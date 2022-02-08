import csv
import os
import shutil

import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, args):
        self.args = args
        self.best_result = 10.0
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        if args.phase == "train":
            self.writer = SummaryWriter(args.log_dir)
            with open(os.path.join(self.args.log_dir, "args.txt"), "w") as log:
                for arg in sorted(vars(self.args)):
                    log.write(
                        arg + ": " + str(getattr(self.args, arg)) + "\n"
                    )  # log of arguments
            self.LOG_FOUT = open(os.path.join(args.log_dir, "log_train.txt"), "a")
            self.fieldnames = ["epoch", "cd", "hd"]
            self.val_csv = os.path.join(args.log_dir, "val.csv")
            with open(self.val_csv, "w") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writeheader()

    def save_info(
        self,
        lr,
        gamma,
        repulsion_loss,
        uniform_loss,
        sparse_loss,
        refine_loss,
        L2_loss,
        hd_loss,
        loss,
        step,
    ):
        self.writer.add_scalar("learning_rate", lr, step)
        self.writer.add_scalar("gamma", gamma, step)
        self.writer.add_scalar("replusion_loss", repulsion_loss, step)
        self.writer.add_scalar("uniform_loss", uniform_loss, step)
        self.writer.add_scalar("sparse_loss", sparse_loss, step)
        self.writer.add_scalar("refine_loss", refine_loss, step)
        self.writer.add_scalar("L2_loss", L2_loss, step)
        self.writer.add_scalar("hd_loss", hd_loss, step)
        self.writer.add_scalar("total_loss", loss, step)

    def log_string(self, msg):
        print(msg)
        self.LOG_FOUT.write(msg + "\n")
        self.LOG_FOUT.flush()

    def print_info(
        self,
        gpu_time,
        total_time,
        sparse_loss,
        refine_loss,
        L2_loss,
        hd_loss,
        loss,
        epoch,
        step,
    ):
        self.log_string("-----------EPOCH %d Step %d:-------------" % (epoch, step))
        self.log_string("  sparse_loss   : {:.6f}".format(sparse_loss))
        self.log_string("  refine_loss   : {:.6f}".format(refine_loss))
        self.log_string("    L2_loss     : {:.6f}".format(L2_loss))
        self.log_string("    hd_loss     : {:.6f}".format(hd_loss))
        self.log_string("     loss       : {:.6f}".format(loss))
        self.log_string("   gpu_time     : {:.6f}".format(gpu_time))
        self.log_string("  total_time    : {:.6f}".format(total_time))

    def save_val_data(self, epoch, cd, hd):
        with open(self.val_csv, "a") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            cd = "{:.5f}".format((cd * 1000))
            hd = "{:.5f}".format((hd * 1000))
            writer.writerow({"epoch": epoch, "cd": cd, "hd": hd})


def adjust_learning_rate(args, epoch, optimizer):
    lr0 = args.base_lr
    ratio = args.lr_decay_ratio
    lr = lr0
    if epoch >= args.start_decay_epoch:
        lr = lr0 * (1 - ratio) ** (
            (epoch - args.start_decay_epoch) // args.lr_decay_epoch
        )
    lr = max(args.lr_clip, lr)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def adjust_gamma(feq, epoch):
    if epoch >= feq * 3:
        gamma = 1.0
    elif epoch >= feq * 2:
        gamma = 0.5
    elif epoch >= feq * 1:
        gamma = 0.1
    else:
        gamma = 0.01
    return gamma


def save_checkpoint(state, is_best, is_save_all, epoch, output_directory):
    checkpoint_filename = os.path.join(
        output_directory, "checkpoint-" + str(epoch) + ".pth.tar"
    )
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, "model_best.pth.tar")
        shutil.copyfile(checkpoint_filename, best_filename)
    if not is_save_all:
        prev_checkpoint_filename = os.path.join(
            output_directory, "checkpoint-" + str(epoch - 1) + ".pth.tar"
        )
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)
