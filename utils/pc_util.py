import os
from glob import glob
from time import time

import numpy as np
import pointnet2_ops.pointnet2_utils as pointnet2
import torch
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def load(filename):
    points = np.loadtxt(filename).astype(np.float32)
    return points


def normalize_point_cloud(input):
    """
    input: pc [N, P, 3]
    output: pc, centroid, furthest_distance
    """
    if len(input.shape) == 2:
        axis = 0
    elif len(input.shape) == 3:
        axis = 1
    centroid = np.mean(input, axis=axis, keepdims=True)
    input = input - centroid
    furthest_distance = np.amax(
        np.sqrt(np.sum(input ** 2, axis=-1, keepdims=True)), axis=axis, keepdims=True
    )
    input = input / furthest_distance
    return input, centroid, furthest_distance


def get_val_data(args):
    print(".....perparing test data.......")
    start = time()
    samples = glob(os.path.join(args.test_dir, "*.xyz"))
    input_val_list = []
    gt_val_list = []
    centroid_val_list = []
    distance_val_list = []
    name_list = [os.path.basename(i) for i in samples]
    n_samples = len(samples)
    for i in range(n_samples):
        input = load(samples[i])
        npoints = input.shape[0]
        input, centroid, furthest_distance = normalize_point_cloud(input)  #
        input_list = []
        input_cuda = input.reshape(1, -1, 3)
        input_cuda = torch.from_numpy(input_cuda).cuda()
        seed_num = int(input.shape[0] * args.patch_num_ratio / args.num_point)
        seed = pointnet2.furthest_point_sample(input_cuda, seed_num)  # 1, seed_num
        seed_coor = input_cuda.squeeze(0)[seed.squeeze(0).long()]
        seed_coor = seed_coor.cpu().numpy()
        seed_sort = np.argsort(
            np.sum((seed_coor[None, :, :] - seed_coor[:, None, :]) ** 2, axis=-1)
        )
        patches = extract_knn_patch(seed_coor, input, args.num_point)
        for j in range(seed_num):
            if args.use_big_patch:
                idx = j
            else:
                idx = find_best_neighbor(patches, seed_sort[j, 1:4], j)
            point = np.stack([patches[j, :], patches[idx, :]])
            input_list.append(point)  # 2 * 256 *3
        if args.gt_dir == "":
            gt_val_list = None
        else:
            gt = load(os.path.join(args.gt_dir, name_list[i]))
            gt, _, _ = normalize_point_cloud(gt)
            gt_val_list.append(torch.from_numpy(gt))
        input_val_list.append(torch.from_numpy(np.vstack(input_list)))
        centroid_val_list.append(torch.from_numpy(centroid))
        distance_val_list.append(torch.from_numpy(furthest_distance))
    print("data is done, It spend : {:.4f} s".format(time() - start))

    return (
        npoints,
        input_val_list,
        gt_val_list,
        centroid_val_list,
        distance_val_list,
        name_list,
    )


def extract_knn_patch(query, pc, k):
    """
    queries [M, C]
    pc [P, C]
    """
    knn_search = NearestNeighbors(n_neighbors=k, algorithm="auto")
    knn_search.fit(pc)
    knn_idx = knn_search.kneighbors(query, return_distance=False)
    k_patches = np.take(pc, knn_idx, axis=0)  # M, K, C
    return k_patches


def find_best_neighbor(patches, idxs, index):
    qpoints = patches[index]
    counts = []
    for i in idxs:
        points = patches[i]
        overlap = np.argwhere((points[:, None, :] == qpoints[:, :]).all(-1))
        overlap = points[overlap[:, 0]]
        if len(overlap):
            db = DBSCAN(eps=0.1, min_samples=10)
            db.fit(overlap)
            # find the overlap
            n_clusters = len(np.unique(db.labels_))
        else:
            n_clusters = 0
        counts.append(n_clusters)
        # get the max
    return idxs[counts.index(max(counts))]


def patch_visualize(lows, highs, out_floder, name):
    low_folder = os.path.join(
        out_floder, os.path.basename(name).split(".")[0], "low_colored_patch"
    )
    high_folder = os.path.join(
        out_floder, os.path.basename(name).split(".")[0], "high_colored_patch"
    )
    folders = [low_folder, high_folder]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
    for i in range(len(lows)):
        # color = get_patch_color(i + 1)
        # color = np.reshape(color, [1, 3])
        color = np.random.randint(0, 256, [1, 3])
        low = lows[i]
        high = np.reshape(highs[i], [-1, 3])

        low_outputs = np.concatenate(
            (low, np.repeat(color, low.shape[0], axis=0)), axis=1
        )
        high_outputs = np.concatenate(
            (high, np.repeat(color, high.shape[0], axis=0)), axis=1
        )

        np.savetxt(
            os.path.join(low_folder, "patch_{}.xyz".format(i)), low_outputs, fmt="%.6f"
        )
        np.savetxt(
            os.path.join(high_folder, "patch_{}.xyz".format(i)),
            high_outputs,
            fmt="%.6f",
        )


def get_patch_color(index):
    Blue = np.array([0, 0, 255], dtype=np.float32)
    Green = np.array([0, 255, 0], dtype=np.float32)
    Yellow = np.array([255, 255, 0], dtype=np.float32)
    Red = np.array([255, 0, 0], dtype=np.float32)
    Colors = [Blue, Green, Yellow, Red]
    first = index // 8
    end = first + 1
    if index % 8 == 0:
        color = Colors[index % 8]
    else:
        delta = Colors[end] - Colors[first]
        radio = (index % 8) / 8
        color = delta * radio + Colors[first]
        color = np.rint(color)
    return color


def normalize_inputs(input):
    centorids = []
    distances = []
    output = []
    num = input.shape[0] // 2
    for i in range(num):
        point = input[2 * i : 2 * (i + 1)]  # 2 * n * 3
        point = point.reshape([-1, 3])
        point, centorid, distance = normalize_point_cloud(point)
        point = point.reshape([2, -1, 3])
        centorids.append(centorid)
        distances.append(distance)
        output.append(point)
        # np.savetxt("{}.xyz".format(i), (point * distance + centorid)[0], fmt="%.6f")
    output = np.vstack(output)
    centorids = np.stack(centorids, 0)
    distances = np.stack(distances, 0)
    output = torch.from_numpy(output)
    centorids = torch.from_numpy(centorids)
    distances = torch.from_numpy(distances)

    return output, centorids, distances
