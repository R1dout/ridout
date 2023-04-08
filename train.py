import csv
import time
import os
from options import prepare_train_args
args = prepare_train_args()
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpus)
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import pickle

from data.data_entry import select_train_loader, select_eval_loader
from model.model_entry import select_model
from utils.logger import Logger
from utils.torch_utils import load_match_dict
from utils.utils import set_cyclic_lr,get_lr

class Trainer:
    def __init__(self):
        self.args = args
        torch.manual_seed(args.seed)
        self.logger = Logger(args)

        self.train_loader = select_train_loader(args)
        self.val_loader = select_eval_loader(args)

        self.model = select_model(args)
        if args.load_model_path != '':
            print("=> using pre-trained weights")
            if args.load_not_strict:
                load_match_dict(self.model, args.load_model_path)
            else:
                self.model.load_state_dict(torch.load(args.load_model_path).state_dict())

        self.model = torch.nn.DataParallel(self.model).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.lr,
                                          betas=(self.args.momentum, self.args.beta),
                                          weight_decay=self.args.weight_decay)
        self.state = {"step" : 0,
             "worse_epochs" : 0,
             "epoch" : 0,
             "best_loss" : np.Inf}

    def train(self):
        print('epoch:{epoch:02d} step:{step:06d}'.format(epoch=self.state['epoch']+1, step=self.state["step"]))
        avg_time = 0.
        self.model.train()
        while self.state['worse_epochs'] < self.args.patience:
            with tqdm(total=len(self.train_loader)) as pbar:
                np.random.seed()
                for i, data in enumerate(self.train_loader):
                    t = time.time()
                    wav, pred, label = self.step(data)
                    set_cyclic_lr(self.optimizer, i, len(self.train_loader) , self.args.cycles, self.args.min_lr,
                                  self.args.lr)
                    self.logger.writer.add_scalar("lr", get_lr(self.optimizer), self.state["step"])
                    metrics = self.compute_metrics(pred, label, is_train=True)
                    loss = metrics['train/'+self.args.loss_fuc]
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.state["step"] += 1
                    t = time.time() - t
                    avg_time += (1. / float(i + 1)) * (t - avg_time)
                    for key in metrics.keys():
                        self.logger.record_scalar(key, metrics[key])
                    pbar.update(1)

            self.model.eval()
            with torch.no_grad():
                total_loss=0
                with tqdm(total=len(self.val_loader)) as pbar:
                    for i, data in enumerate(self.val_loader):
                        wav, pred, label = self.step(data)
                        metrics = self.compute_metrics(pred, label, is_train=False)
                        loss = metrics['val/'+self.args.loss_fuc]
                        total_loss += loss
                        for key in metrics.keys():
                            self.logger.record_scalar(key, metrics[key])
                        pbar.update(1)
                    ave_loss = total_loss/(i+1)
                    print("VALIDATION FINISHED: LOSS: " + str(ave_loss))
                    if ave_loss >= self.state["best_loss"]:
                        self.state["worse_epochs"] += 1
                    else:
                        print("MODEL IMPROVED ON VALIDATION SET!")
                        self.state["worse_epochs"] = 0
                        self.state["best_loss"] = ave_loss
                        self.state["best_checkpoint"] = '{epoch:02d}_{step:06d}.pth'.format(epoch=self.state['epoch'],step=self.state["step"])

                    self.state['epoch']+=1
                    self.logger.save_curves(self.state['epoch'])
                    self.logger.save_check_point(self.model,self.state['epoch'],self.state["step"],self.args.gpus)
                    self.save_state(self.args.model_dir,self.state)

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
    
    def save_state(self,model_dir,state):
        with open(model_dir + '/state.txt', 'wb') as file:
            file.write(pickle.dumps(str(state)))

def main():
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()
