import os
import random
from time import time

import numpy as np
import pointnet2_ops.pointnet2_utils as pointnet2
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import utils.pc_util as pc_util
from common.configs import args
from common.data_loader import Dataset, PUGAN_Dataset
from common.helper import Logger, adjust_gamma, adjust_learning_rate, save_checkpoint
from common.loss import Loss
from network.model import Model


def xavier_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.xavier_normal_(m.weight)
    elif classname.find("Linear") != -1:
        torch.nn.init.xavier_normal_(m.weight)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


def test(
    model,
    npoints,
    input_val_list,
    gt_val_list,
    centroid_val_list,
    distance_val_list,
    name_list,
):
    start = time()
    val_loss = Loss()
    cd_loss, hd_loss = 0.0, 0.0
    num_sample = len(input_val_list)
    out_folder = os.path.join(args.log_dir, args.out_dir)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    with torch.no_grad():
        for i in range(num_sample):
            torch.cuda.empty_cache()
            d = distance_val_list[i].float().to(device)
            c = centroid_val_list[i].float().to(device)
            # input [n * 2, 256, 3], gt [8192, 3] torch.tensor, centroid 1 distance [3]
            input_list = input_val_list[i]  # n * 2 256 3
            gt = gt_val_list[i]
            gt = gt.unsqueeze(0).float().to(device)
            input, centorids, furthest_distances = pc_util.normalize_inputs(
                input_list.numpy()
            )
            input = input.permute(0, 2, 1).contiguous().float().to(device)
            _, pred = model(input)
            pred = pred[::2, :, :]  # n 3 256
            pred = pred.permute(0, 2, 1).contiguous()  # n 256 3
            input_list = input_list[::2, :, :]  # n 256 3
            pred = pred * furthest_distances.to(device) + centorids.to(device)
            if args.patch_visualize and args.phase == "test":
                pc_util.patch_visualize(
                    input_list.numpy(), pred.cpu().numpy(), out_folder, name_list[i],
                )
            pred = pred * d + c  # 去归一化
            pred = pred.reshape([1, -1, 3])
            index = pointnet2.furthest_point_sample(pred, npoints * args.up_ratio)
            pred = pred.squeeze(0)[index.squeeze(0).long()]
            np.savetxt(
                os.path.join(out_folder, name_list[i]), pred.cpu().numpy(), fmt="%.6f"
            )

            # 针对上采样不同倍数设计
            if pred.shape[0] == gt.shape[1]:
                pred, _, _ = pc_util.normalize_point_cloud(pred.cpu().numpy())
                pred = torch.from_numpy(pred).unsqueeze(0).contiguous().to(device)
                # 1 n 3
                cd_loss += val_loss.get_cd_loss(pred, gt).item()
                hd_loss += val_loss.get_hd_loss(pred, gt).item()
            else:
                cd_loss = 1.0
                hd_loss = 1.0

    print(
        "cd loss : {:.4f}, hd loss : {:.4f}".format(
            (cd_loss / num_sample * 1000), (hd_loss / num_sample * 1000)
        )
    )
    print("测试共花费:{:.6f}s".format(time() - start))
    del d, c, input, gt, pred, index  # 清楚内存
    torch.cuda.empty_cache()

    return cd_loss / num_sample, hd_loss / num_sample


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


setup_seed(args.seed)
device = torch.device("cuda")
Loss_fn = Loss()

checkpoint_path = os.path.join(args.log_dir, args.checkpoint_path)
checkpoint = None
lr = args.base_lr
logger = Logger(args)
# 定义日志
(
    npoints,
    input_val_list,
    gt_val_list,
    centroid_val_list,
    distance_val_list,
    name_list,
) = pc_util.get_val_data(args)
if args.phase == "test" or args.restore:
    print("=> loading checkpoint '{}' ... ".format(checkpoint_path), end="")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    start_epoch = checkpoint["epoch"] + 1
    logger.best_result = checkpoint["best_result"]
else:
    start_epoch = 0
model = Model(args).to(device)
model_named_params = [p for _, p in model.named_parameters() if p.requires_grad]
optimizer = torch.optim.Adam(model_named_params, lr=lr, betas=(args.beta1, args.beta2))

if checkpoint:
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print("=> checkpoint state loaded.")
else:
    model.apply(xavier_init)  # 参数初始化
model = torch.nn.DataParallel(model)
# 多GPU训练
if args.phase == "train":
    if args.use_single_patch:
        train_dataset = PUGAN_Dataset(args)
        train_data_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        train_dataset = Dataset(args)
        train_data_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size // 2,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    n_set = len(train_data_loader)
    for epoch in range(start_epoch, args.training_epoch):
        model.train()
        is_best = False
        lr = adjust_learning_rate(args, epoch, optimizer)
        gamma = adjust_gamma(args.fidelity_feq, epoch)
        # 调整学习率
        for idx, (input, gt, radius) in enumerate(train_data_loader):
            start = time()
            optimizer.zero_grad()
            # B N 3
            input = input.reshape(-1, args.num_point, 3)
            gt = gt.reshape(-1, args.num_point * args.up_ratio, 3)
            radius = radius.reshape(-1)
            input = input.permute(0, 2, 1).contiguous().float().to(device)
            gt = gt.permute(0, 2, 1).contiguous().float().to(device)
            radius = radius.float().to(device)
            gpu_start = time()
            sparse, refine = model(input)
            gpu_time = time() - gpu_start
            sparse = sparse.permute(0, 2, 1).contiguous()
            refine = refine.permute(0, 2, 1).contiguous()
            gt = gt.permute(0, 2, 1).contiguous()

            if args.use_repulse:
                repulsion_loss = args.repulsion_w * Loss_fn.get_repulsion_loss(refine)
            else:
                repulsion_loss = torch.tensor(0.0)
            if args.use_uniform:
                uniform_loss = args.uniform_w + Loss_fn.get_uniform_loss(
                    refine, radius=radius
                )
            else:
                uniform_loss = torch.tensor(0.0)
            if args.use_l2:
                L2_loss = Loss_fn.get_l2_regular_loss(model, args.regular_w)
            else:
                L2_loss = torch.tensor(0.0)
            if args.use_hd:
                hd_loss = args.hd_w * Loss_fn.get_hd_loss(refine, gt, radius)
            else:
                hd_loss = torch.tensor(0.0)
            if args.use_emd:
                sparse_loss = args.fidelity_w * Loss_fn.get_emd_loss(sparse, gt, radius)
                refine_loss = args.fidelity_w * Loss_fn.get_emd_loss(refine, gt, radius)
            else:
                sparse_loss = args.fidelity_w * Loss_fn.get_cd_loss(sparse, gt, radius)
                refine_loss = args.fidelity_w * Loss_fn.get_cd_loss(refine, gt, radius)

            loss = (
                gamma * refine_loss
                + gamma * hd_loss
                + sparse_loss
                + repulsion_loss
                + uniform_loss
                + L2_loss
            )
            loss.backward()
            optimizer.step()

            step = epoch * n_set + idx
            logger.save_info(
                lr,
                gamma,
                repulsion_loss.item(),
                uniform_loss.item(),
                sparse_loss.item(),
                refine_loss.item(),
                L2_loss.item(),
                hd_loss.item(),
                loss.item(),
                step,
            )  # 写入tensorboard
            total_time = time() - start
            logger.print_info(
                gpu_time,
                total_time,
                sparse_loss.item(),
                refine_loss.item(),
                L2_loss.item(),
                hd_loss.item(),
                loss.item(),
                epoch,
                step,
            )  # 打印
        if epoch > args.start_eval_epoch:
            # epoch大于40之后再开始求最好的
            model.eval()
            cd, hd = test(
                model,
                npoints,
                input_val_list,
                gt_val_list,
                centroid_val_list,
                distance_val_list,
                name_list,
            )
            logger.save_val_data(epoch, cd, hd)
            if logger.best_result > cd:
                logger.best_result = cd
                is_best = True
        save_checkpoint(
            {
                "epoch": epoch,
                "model": model.module.state_dict(),
                "best_result": logger.best_result,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            epoch,
            args.log_dir,
        )  #

else:
    model.eval()
    test(
        model,
        npoints,
        input_val_list,
        gt_val_list,
        centroid_val_list,
        distance_val_list,
        name_list,
    )
    # 4倍上采样
    if args.n_upsample == 2:
        args.test_dir = os.path.join(args.log_dir, args.out_dir)
        args.out_dir = "{}_up_2".format(args.out_dir)
        args.gt_dir = "data/test/gt_32768"
        (
            npoints,
            input_val_list,
            gt_val_list,
            centroid_val_list,
            distance_val_list,
            name_list,
        ) = pc_util.get_val_data(args)
        test(
            model,
            npoints,
            input_val_list,
            gt_val_list,
            centroid_val_list,
            distance_val_list,
            name_list,
        )
    # 16倍上采样

