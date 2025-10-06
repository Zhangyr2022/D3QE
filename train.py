import os
import time
from tensorboardX import SummaryWriter
from tqdm import tqdm

from validate import validate
from data import create_dataloader_new
from networks.trainer import Trainer
from options import TrainOptions
from data.process import get_processing_model
from util import set_random_seed, EarlyStopping

"""Currently assumes jpg_prob, blur_prob 0 or 1"""


def get_val_opt():
    """Build a validation option object derived from training options."""
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = "{}/{}/".format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.isVal = True
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.jpg_method = ["pil"]
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]
    return val_opt


def setup_writers(opt):
    """Create TensorBoard writers for train and validation logs."""
    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
    return train_writer, val_writer


def run_validation(model, val_opt, val_writer, global_step, tag_prefix=""):
    """Run validation, log metrics to TensorBoard, and print a short summary."""
    acc, ap = validate(model, val_opt)[:2]
    if val_writer is not None:
        val_writer.add_scalar("accuracy", acc, global_step)
        val_writer.add_scalar("ap", ap, global_step)
    print(
        "(Val {}{}) acc: {}; ap: {}".format(
            tag_prefix, val_opt.isVal and "" or "", acc, ap
        )
    )
    return acc, ap


def train_one_epoch(model, data_loader, opt, train_writer, val_writer, val_opt, epoch):
    """Train for a single epoch with progress bar and periodic validation/saves."""
    with tqdm(total=len(data_loader), desc=f"Training Epoch {epoch}") as pbar:
        for i, data in enumerate(data_loader):
            if opt.epoch_counter is not None and opt.epoch_counter < i:
                break

            model.total_steps += 1
            model.set_input(data)
            model.optimize_parameters()

            pbar.set_postfix_str(f"loss: {model.loss:.4f}")
            pbar.update(1)

            if model.total_steps % opt.loss_freq == 0:
                print(
                    "Train loss: {} at step: {}".format(model.loss, model.total_steps)
                )
                train_writer.add_scalar("loss", model.loss, model.total_steps)

            if model.total_steps % opt.save_latest_freq == 0:
                print(
                    "saving the latest model {} (epoch {}, model.total_steps {})".format(
                        opt.name, epoch, model.total_steps
                    )
                )
                model.save_networks("latest")
                model.eval()
                acc, ap = validate(model.model, val_opt)[:2]
                val_writer.add_scalar("accuracy", acc, model.total_steps)
                val_writer.add_scalar("ap", ap, model.total_steps)
                print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))
                model.train()


def main():
    """Entry point for training loop with early stopping and periodic evaluation."""
    set_random_seed()
    opt = TrainOptions().parse()
    opt.dataroot = "{}/{}/".format(opt.dataroot, opt.train_split)
    val_opt = get_val_opt()

    data_loader = create_dataloader_new(opt)
    dataset_size = len(data_loader)
    print("#training images = %d" % dataset_size)

    train_writer, val_writer = setup_writers(opt)

    model = Trainer(opt)
    early_stopping = EarlyStopping(
        patience=opt.earlystop_epoch, delta=-0.001, verbose=True
    )

    opt = get_processing_model(opt)

    for epoch in range(opt.niter):
        _ = time.time()  # epoch_start_time; kept minimal to avoid unused variable
        _ = time.time()  # iter_data_time; kept minimal to avoid unused variable

        train_one_epoch(
            model, data_loader, opt, train_writer, val_writer, val_opt, epoch
        )

        print("break")  # preserved from original behavior

        if epoch % opt.save_epoch_freq == 0:
            print(
                "saving the model at the end of epoch %d, iters %d"
                % (epoch, model.total_steps)
            )
            model.save_networks("latest")
            model.save_networks(epoch)

        # Validation at end of epoch
        model.eval()
        acc, ap = validate(model.model, val_opt)[:2]
        val_writer.add_scalar("accuracy", acc, model.total_steps)
        val_writer.add_scalar("ap", ap, model.total_steps)
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))

        early_stopping(acc, model)

        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(
                    patience=opt.earlystop_epoch, delta=-0.002, verbose=True
                )
            else:
                print("Early stopping.")
                break

        model.train()


if __name__ == "__main__":
    main()
