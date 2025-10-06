import os
import torch
from collections import OrderedDict
import argparse
import numpy as np
import random
from networks.D3QE import D3QE

# def diffusion_defaults():
#     """
#     Defaults for image and classifier training.
#     """
#     return dict(
#         learn_sigma=True,
#         diffusion_steps=1000,
#         noise_schedule="linear",
#         timestep_respacing="ddim20",
#         use_kl=False,
#         predict_xstart=False,
#         rescale_timesteps=False,
#         rescale_learned_sigmas=False,
#     )


# def model_and_diffusion_defaults():
#     """
#     Defaults for image training.
#     """
#     res = dict(
#         image_size=256,
#         num_channels=256,
#         num_res_blocks=2,
#         num_heads=4,
#         num_heads_upsample=-1,
#         num_head_channels=64,
#         attention_resolutions="32,16,8",
#         channel_mult="",
#         dropout=0.1,
#         class_cond=False,
#         use_checkpoint=False,
#         use_scale_shift_norm=True,
#         resblock_updown=True,
#         use_fp16=True,
#         use_new_attention_order=False,
#     )

#     res.update(diffusion_defaults())
#     return res


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def unnormalize(tens, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # assume tensor of shape NxCxHxW
    return (
        tens * torch.Tensor(std)[None, :, None, None]
        + torch.Tensor(mean)[None, :, None, None]
    )


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def create_argparser():
    defaults = dict(
        batch_size=1,
        model_path="/path/to/your/model",
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def get_model(opt):
    if opt.detect_method == "D3QE":
        return D3QE(opt.vqvae_path)
    else:
        raise ValueError(f"Unsupported model_type: {opt.detect_method}")


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=1, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.score_max = -np.inf
        self.delta = delta

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(
                f"Validation accuracy increased ({self.score_max:.6f} --> {score:.6f}).  Saving model ..."
            )
        model.save_networks("best")
        self.score_max = score
