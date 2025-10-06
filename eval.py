"""
Usage:
python eval_all.py --model_path ./weights/{}.pth --detect_method {CNNSpot,Gram,Fusing,FreDect,LGrad,LNP,DIRE} --noise_type {blur,jpg,resize}
"""

import os
import csv
import torch
import multiprocessing

from PIL import ImageFile
from validate import validate
from options import TestOptions
from util import create_argparser, get_model, set_random_seed


def load_model(opt):
    """
    Load model weights and return the model instance.
    """
    model = get_model(opt)
    state_dict = torch.load(opt.model_path, map_location="cuda")
    try:
        model.load_state_dict(state_dict["model"], strict=True)
    except Exception as e:
        info = model.load_state_dict(state_dict["model"], strict=False)
        print(f"Warning: {e}\nPartial state_dict loaded: {info}")
    model.cuda()
    model.eval()
    return model


def eval_model(model, opt, rows):
    """
    Evaluate the model on all validation sets specified in opt.vals.
    Appends results to the provided rows list.
    Returns average accuracy and average precision.
    """
    avg_acc, avg_ap = 0, 0
    cur_dataroot = opt.dataroot
    cur_vals = opt.sub_dir
    for val in cur_vals:
        opt.dataroot = os.path.join(cur_dataroot, val)
        opt.process_device = torch.device("cuda")
        acc, ap, r_acc, f_acc, _, _ = validate(model, opt)
        rows.append([val, acc, ap, r_acc, f_acc])
        print(
            f"[{val}] Accuracy: {acc:.4f} | AP: {ap:.4f} | r_acc: {r_acc:.4f} | f_acc: {f_acc:.4f}"
        )
        avg_acc += acc
        avg_ap += ap
    avg_acc /= len(cur_vals)
    avg_ap /= len(cur_vals)
    return avg_acc, avg_ap


def save_results(rows, csv_name):
    """
    Save evaluation results to a CSV file.
    """
    with open(csv_name, "a+") as f:
        csv_writer = csv.writer(f, delimiter=",")
        csv_writer.writerows(rows)


def eval_with_disturbance(model, opt, results_dir, model_name):
    """
    Evaluate model robustness under different noise settings.
    """
    # JPEG compression robustness
    for jpg_qual in [95, 90, 75, 60, 50]:
        opt.jpg_qual = [str(jpg_qual)]
        opt.noise_type = "jpg"
        print(f"\n{'-'*20}\nEvaluating JPEG quality: {jpg_qual}\n{'-'*20}")
        rows = [
            [f"{model_name} model testing on... jpg {jpg_qual}"],
            ["testset", "accuracy", "avg precision", "r_acc", "f_acc"],
        ]
        avg_acc, avg_ap = eval_model(model, opt, rows)
        rows.append(["average", avg_acc, avg_ap, "-", "-"])
        csv_name = os.path.join(
            results_dir, f"{opt.detect_method}_{opt.noise_type}_{jpg_qual}_correct.csv"
        )
        save_results(rows, csv_name)

    # Cropping robustness
    for crop_ratio in [0.9, 0.8, 0.7, 0.6, 0.5]:
        opt.crop_ratio = crop_ratio
        opt.noise_type = "crop"
        print(f"\n{'-'*20}\nEvaluating crop ratio: {crop_ratio}\n{'-'*20}")
        rows = [
            [f"{model_name} model testing on... crop {crop_ratio}"],
            ["testset", "accuracy", "avg precision", "r_acc", "f_acc"],
        ]
        avg_acc, avg_ap = eval_model(model, opt, rows)
        rows.append(["average", avg_acc, avg_ap, "-", "-"])
        csv_name = os.path.join(
            results_dir, f"{opt.detect_method}_{opt.noise_type}_{crop_ratio}.csv"
        )
        save_results(rows, csv_name)


def main():
    """
    Main entry point for evaluation.
    """
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    set_random_seed()

    opt = TestOptions().parse(print_options=True)
    model_name = os.path.basename(opt.model_path).replace(".pth", "")
    results_dir = os.path.join("./results", opt.detect_method)
    os.makedirs(results_dir, exist_ok=True)

    print(f"Starting evaluation for model: {model_name}")

    model = load_model(opt)

    if opt.robust_all:
        eval_with_disturbance(model, opt, results_dir, model_name)
    else:
        rows = [
            [f"{model_name} model testing on..."],
            ["testset", "accuracy", "avg precision", "r_acc", "f_acc"],
        ]
        print(f"Evaluation options: {opt}")
        avg_acc, avg_ap = eval_model(model, opt, rows)
        rows.append(["average", avg_acc, avg_ap, "-", "-"])
        csv_name = os.path.join(results_dir, f"{opt.detect_method}_{opt.detail}.csv")
        save_results(rows, csv_name)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
