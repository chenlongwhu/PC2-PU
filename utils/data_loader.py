import h5py
import torch.utils.data as data
import sys

sys.path.append("../")
import numpy as np
import utils.data_util as utils


def load_h5_data(filename):
    f = h5py.File(filename, "r")
    input = f["poisson_pair"][:]
    gt = f["poisson_pair"][:]
    return input, gt


class Dataset(data.Dataset):
    def __init__(self, args):
        super().__init__()
        h5_path = args.data_dir
        # 9000 * 2048 * 3
        self.input, self.gt = load_h5_data(h5_path)
        self.radius = np.ones(shape=(len(self.input)))
        self.data_npoint = args.num_point * args.up_ratio
        self.npoint = args.num_point
        self.augment = args.augment
        self.jitter_sigma = args.jitter_sigma
        self.jitter_max = args.jitter_max

        centroid = np.mean(self.gt[:, :, 0:3], axis=1, keepdims=True)
        # 算出每一个的平均值
        self.gt[:, :, 0:3] = self.gt[:, :, 0:3] - centroid
        # 中心化
        furthest_distance = np.amax(
            np.sqrt(np.sum(self.gt[:, :, 0:3] ** 2, axis=-1)), axis=1, keepdims=True
        )
        self.gt[:, :, 0:3] = self.gt[:, :, 0:3] / np.expand_dims(
            furthest_distance, axis=-1
        )
        self.input[:, :, 0:3] = self.input[:, :, 0:3] - centroid
        self.input[:, :, 0:3] = self.input[:, :, 0:3] / np.expand_dims(
            furthest_distance, axis=-1
        )
        # 归一化
        print("total %d samples" % (len(self.input)))

    def __load_split_file(self):
        index = np.loadtxt(self.split_dir)
        index = index.astype(np.int)
        print(index)
        self.input = self.input[index, :]
        self.gt = self.gt[index, :]
        self.radius = self.radius[index]

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, index):
        input_data = self.input[index]
        gt_data = self.gt[index]
        radius_data = np.array([self.radius[index]])
        input_data = input_data.reshape(2, -1, 3)
        gt_data = gt_data.reshape(2, -1, 3)
        radius_data = np.tile(radius_data, 2)
        sample_idx1 = utils.nonuniform_sampling(
            self.data_npoint, sample_num=self.npoint
        )
        sample_idx2 = utils.nonuniform_sampling(
            self.data_npoint, sample_num=self.npoint
        )
        input_data = np.stack((input_data[0][sample_idx1], input_data[1][sample_idx2]))
        if self.augment:
            # for data aug
            input_data = utils.jitter_perturbation_point_cloud(
                input_data, sigma=self.jitter_sigma, clip=self.jitter_max
            )
            input_data, gt_data = utils.rotate_point_cloud_and_gt(input_data, gt_data)
            input_data, gt_data, scale = utils.random_scale_point_cloud_and_gt(
                input_data, gt_data, scale_low=0.9, scale_high=1.1
            )
            radius_data = radius_data * scale

        return input_data, gt_data, radius_data


if __name__ == "__main__":
    from utils.configs import args
    from torch.utils.data import DataLoader

    dataset = Dataset(args)
    train_data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size // 2,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    for i, (input, gt, radius) in enumerate(train_data_loader):
        print(input[0])
        print(input.reshape(args.batch_size, -1, 3).shape)
        print(input.shape, gt.shape, radius.shape)
    # (input_data,gt_data,radius_data)=dataset.__getitem__(0)
    # print(input_data.shape,gt_data.shape,radius_data.shape)
    # dataset=PUNET_Dataset_Whole(data_dir="../MC_5k",n_input=1024)
    # points=dataset.__getitem__(0)
    # print(points.shape)

