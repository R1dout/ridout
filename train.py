import csv
import time

import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.autograd import Variable

from data.data_entry import select_train_loader, select_eval_loader
from model.model_entry import select_model
from options import prepare_train_args
from utils.logger import Logger
from utils.torch_utils import load_match_dict


class Trainer:
    def __init__(self):
        args = prepare_train_args()
        self.args = args
        torch.manual_seed(args.seed)
        self.logger = Logger(args)

        self.train_loader = select_train_loader(args)
        self.val_loader = select_eval_loader(args)

        self.model = select_model(args)
        if args.load_model_path != '':
            print("=> using pre-trained weights for DPSNet")
            if args.load_not_strict:
                load_match_dict(self.model, args.load_model_path)
            else:
                self.model.load_state_dict(torch.load(args.load_model_path).state_dict())

        self.model = torch.nn.DataParallel(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.lr,
                                          betas=(self.args.momentum, self.args.beta),
                                          weight_decay=self.args.weight_decay)

    def train(self):
        for epoch in range(self.args.epochs):
            # train for one epoch
            self.train_per_epoch(epoch)
            self.val_per_epoch(epoch)
            self.logger.save_curves(epoch)
            self.logger.save_check_point(self.model, epoch)

    def train_per_epoch(self, epoch):
        # switch to train mode
        self.model.train()

        for i, data in enumerate(self.train_loader):
            wav, pred, label = self.step(data)

            # compute loss
            metrics = self.compute_metrics(pred, label, is_train=True)

            # get the item for backward
            loss = metrics['train/l1']

            # compute gradient and do Adam step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # logger record
            for key in metrics.keys():
                self.logger.record_scalar(key, metrics[key])

            # monitor training progress
            if i % self.args.print_freq == 0:
                print('Train: Epoch {} batch {} Loss {}'.format(epoch, i, loss))

    def val_per_epoch(self, epoch):
        self.model.eval()
        for i, data in enumerate(self.val_loader):
            wav, pred, label = self.step(data)
            metrics = self.compute_metrics(pred, label, is_train=False)

            for key in metrics.keys():
                self.logger.record_scalar(key, metrics[key])

    def step(self, data):
        wav, label = data
        # warp input
        wav = Variable(wav).cuda()
        label = Variable(label).cuda()

        # compute output
        pred = self.model(wav)
        return wav, pred, label

    def compute_metrics(self, pred, gt, is_train):
        # you can call functions in metrics.py
        l1 = (pred - gt).abs().mean()
        prefix = 'train/' if is_train else 'val/'
        metrics = {
            prefix + 'l1': l1
        }
        return metrics


def main():
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()
