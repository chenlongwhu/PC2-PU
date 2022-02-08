import argparse
import csv
import os
import subprocess
from collections import OrderedDict
from glob import glob
from time import time

import numpy as np
import torch

from common.loss import Loss
from utils.pc_util import load, normalize_point_cloud


def compute_p2f(folder):
    for name in gt_names:
        mesh_path = os.path.join(MESH_DIR, name[:-4] + ".off")
        point_path = os.path.join(folder, name[:-4] + ".xyz")
        commands = ["./evaluate", mesh_path, point_path]
        subprocess.run(commands, stdout=devNull)
    print("compute done")


parser = argparse.ArgumentParser()
parser.add_argument("--pred", nargs="+", type=str)
parser.add_argument("--gt", type=str, required=True, help=".xyz")
parser.add_argument("--out_folder", type=str, required=True)
FLAGS = parser.parse_args()
print(FLAGS.pred)
PRED_DIR = [os.path.abspath(p) for p in FLAGS.pred]
GT_DIR = os.path.abspath(FLAGS.gt)
MESH_DIR = os.path.abspath("data/mesh")

if not os.path.exists(FLAGS.out_folder):
    os.makedirs(FLAGS.out_folder)
loss = Loss()

gt_paths = glob(os.path.join(GT_DIR, "*.xyz"))
gt_names = [os.path.basename(p) for p in gt_paths]
gt_data = [load(p) for p in gt_paths]
gt_data = np.stack(gt_data)
gt_data, _, _ = normalize_point_cloud(gt_data)

fieldnames = ["name", "CD", "HD", "p2f avg", "p2f std"]
counter = len(gt_paths)
devNull = open(os.devnull, "w")

for D in PRED_DIR:
    name = gt_names[0]
    pred_path = os.path.join(D, name)
    if os.path.isfile(pred_path[:-4] + "_point2mesh_distance.txt"):
        break
    else:
        compute_p2f(D)

start = time()
for D in PRED_DIR:
    avg_cd_value = 0
    avg_hd_value = 0
    global_p2f = []

    csv_file = os.path.join(FLAGS.out_folder, "{}_eval.csv".format(D.split("/")[-1]))
    with open(csv_file, "w") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, restval="-", extrasaction="ignore"
        )
        writer.writeheader()
        for i, name in enumerate(gt_names):
            row = {}
            gt = gt_data[i]
            pred_path = os.path.join(D, name)
            if not os.path.isfile(pred_path):
                break
            pred = load(pred_path)
            pred, _, _ = normalize_point_cloud(pred)

            gt = torch.from_numpy(gt).unsqueeze(0).contiguous().cuda()
            pred = torch.from_numpy(pred).unsqueeze(0).contiguous().cuda()
            cd_loss = 1000.0 * loss.get_cd_loss(pred, gt).cpu().item()
            hd_loss = 1000.0 * loss.get_hd_loss(pred, gt).cpu().item()
            avg_cd_value += cd_loss
            avg_hd_value += hd_loss
            row["name"] = name[:-4]
            row["CD"] = "{:.6f}".format(cd_loss)
            row["HD"] = "{:.6f}".format(hd_loss)
            if os.path.isfile(pred_path[:-4] + "_point2mesh_distance.txt"):
                point2mesh_distance = load(pred_path[:-4] + "_point2mesh_distance.txt")
                if point2mesh_distance.size == 0:
                    continue
                point2mesh_distance = point2mesh_distance[:, 3]
                row["p2f avg"] = "{:.6f}".format(np.nanmean(point2mesh_distance))
                row["p2f std"] = "{:.6f}".format(np.nanstd(point2mesh_distance))
                global_p2f.append(point2mesh_distance)
            writer.writerow(row)
        row = OrderedDict()
        avg_cd_value /= counter
        avg_hd_value /= counter
        row["name"] = "average"
        row["CD"] = "{:.6f}".format(avg_cd_value)
        row["HD"] = "{:.6f}".format(avg_hd_value)
        if global_p2f:
            global_p2f = np.concatenate(global_p2f, axis=0)
            mean_p2f = np.nanmean(global_p2f)
            std_p2f = np.nanstd(global_p2f)
            row["p2f avg"] = "{:.6f}".format(mean_p2f)
            row["p2f std"] = "{:.6f}".format(std_p2f)
        writer.writerow(row)

print("It spend :{:.4f}s".format(time() - start))
print("done")

