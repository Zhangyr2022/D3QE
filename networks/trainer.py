import functools
import torch
import torch.nn as nn

from networks.base_model import BaseModel, init_weights

from util import get_model


class Trainer(BaseModel):
    def name(self):
        return "Trainer"

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        self.model = get_model(opt)

        if self.isTrain:
            print("Using BCEWithLogitsLoss")
            self.loss_fn = nn.BCEWithLogitsLoss()
            # initialize optimizers
            params = self.model.parameters()
            if opt.optim == "adam":
                print("Using AdamW")
                self.optimizer = torch.optim.AdamW(
                    params,
                    lr=opt.lr,
                    betas=(opt.beta1, 0.999),
                    weight_decay=opt.weight_decay,
                )
            elif opt.optim == "sgd":
                self.optimizer = torch.optim.SGD(
                    params, lr=opt.lr, momentum=0.0, weight_decay=0
                )
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        self.model.to(self.device)

    def adjust_learning_rate(self, min_lr=1e-6, step=10):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] /= step
            if param_group["lr"] < min_lr:
                return False
        print("Learning rate adjusted to: {}".format(param_group["lr"]))
        return True

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()

    def forward(self):
        if self.opt.detect_method == "D3QE":
            self.output = self.model(self.input, self.label)
        else:
            raise ValueError(f"Unsupported detect method: {self.opt.detect_method}")

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        # with torch.autograd.detect_anomaly():
        self.loss.backward()
        self.optimizer.step()
