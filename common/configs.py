import argparse


def str2bool(x):
    return x.lower() in ("true")


parser = argparse.ArgumentParser()
parser.add_argument("--phase", default="train", help="train/test")
parser.add_argument("--log_dir", default="log/PC2-PU")
parser.add_argument("--data_dir", default="data/train/PC2-PU.h5")
parser.add_argument("--test_dir", default="data/test/gt_2048")
parser.add_argument("--gt_dir", default="data/test/gt_8192")
parser.add_argument("--out_dir", default="4倍上采样测试")
parser.add_argument("--checkpoint_path", default="model_best.pth.tar")
parser.add_argument("--restore", action="store_true")
parser.add_argument("--n_upsample", type=int, default=1)
parser.add_argument("--training_epoch", type=int, default=400)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--augment", type=str2bool, default=True)
parser.add_argument("--jitter", type=str2bool, default=False)
parser.add_argument(
    "--jitter_sigma", type=float, default=0.01, help="jitter augmentation"
)
parser.add_argument(
    "--jitter_max", type=float, default=0.03, help="jitter augmentation"
)
parser.add_argument("--K", type=int, default=16)
parser.add_argument("--K1", type=int, default=20)
parser.add_argument("--K2", type=int, default=20)
parser.add_argument("--transform_dim", type=int, default=64)
parser.add_argument("--up_ratio", type=int, default=4)
parser.add_argument("--num_point", type=int, default=256)
parser.add_argument("--patch_num_ratio", type=int, default=3)
parser.add_argument("--base_lr", type=float, default=0.001)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.999)
parser.add_argument("--start_decay_epoch", type=int, default=20)
parser.add_argument("--lr_decay_epoch", type=int, default=20)
parser.add_argument("--lr_decay_ratio", type=float, default=0.1)
parser.add_argument("--lr_clip", type=float, default=1e-6)
parser.add_argument("--seed", type=int, default=2021)
parser.add_argument("--start_eval_epoch", type=int, default=40)

parser.add_argument("--use_repulse", type=str2bool, default=False)
parser.add_argument("--use_emd", type=str2bool, default=True)
parser.add_argument("--use_hd", type=str2bool, default=False)
parser.add_argument("--use_uniform", type=str2bool, default=False)
parser.add_argument("--use_l2", type=str2bool, default=True)
parser.add_argument("--use_big_patch", type=str2bool, default=False)
parser.add_argument("--is_save_all", type=str2bool, default=False)
parser.add_argument("--up_module", type=str, default="shuffle")

parser.add_argument("--repulsion_w", default=1.0, type=float, help="repulsion_weight")
parser.add_argument("--uniform_w", default=10.0, type=float, help="uniform_weight")
parser.add_argument("--patch_visualize", type=str2bool, default=True)
parser.add_argument("--fidelity_feq", type=int, default=10)
parser.add_argument("--fidelity_w", type=float, default=100.0)
parser.add_argument("--hd_w", type=float, default=10.0)
parser.add_argument("--regular_w", type=float, default=1e-5)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--gpu", nargs="+", type=str, default="0")

args = parser.parse_args()
