import sys

import h5py
import torch.utils.data as data

sys.path.append("../")
import numpy as np
import utils.data_util as utils
from utils.pc_util import normalize_point_cloud


def load_h5_data(filename):
    f = h5py.File(filename, "r")
    input = f["poisson_pair"][:]
    gt = f["poisson_pair"][:]
    return input, gt


def load_pugan_h5_data(filename):
    f = h5py.File(filename, "r")
    input = f["poisson_2048"][:]
    gt = f["poisson_2048"][:]
    data_radius = np.ones(shape=(len(input)))
    gt, centroid, furthest_distance = normalize_point_cloud(gt)
    input[:, :, 0:3] = input[:, :, 0:3] - centroid
    input[:, :, 0:3] = input[:, :, 0:3] / furthest_distance

    return input, gt, data_radius


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
        self.gt[:, :, 0:3] = self.gt[:, :, 0:3] - centroid
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
        print("total %d samples" % (len(self.input)))

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


class PUGAN_Dataset(data.Dataset):
    def __init__(self, args):
        super().__init__()
        h5_path = "data/train/MYNET_big_patch_{}.h5".format(args.num_point * 4)
        self.input, self.gt, self.radius = load_pugan_h5_data(h5_path)
        self.data_npoint = args.num_point * args.up_ratio
        self.npoint = args.num_point
        self.augment = args.augment
        self.jitter_sigma = args.jitter_sigma
        self.jitter_max = args.jitter_max

        print("total %d samples" % (len(self.input)))

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, index):
        input_data = self.input[index]
        gt_data = self.gt[index]
        radius_data = np.array([self.radius[index]])

        sample_idx = utils.nonuniform_sampling(self.data_npoint, sample_num=self.npoint)
        input_data = input_data[sample_idx, :]

        input_data = np.expand_dims(input_data, axis=0)
        gt_data = np.expand_dims(gt_data, axis=0)
        if self.augment:
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
    from torch.utils.data import DataLoader
    from configs import args

    args.data_dir = "../data/train/PC2-PU.h5"

    dataset = PUGAN_Dataset(args)
    train_data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    for i, (input, gt, radius) in enumerate(train_data_loader):
        print(input.shape, gt.shape, radius.shape)
        exit()
