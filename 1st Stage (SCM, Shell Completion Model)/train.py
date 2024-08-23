import warnings

#warnings.simplefilter("ignore", (UserWarning, FutureWarning))
from utils.hparams import HParam
from torchvision import transforms
from tqdm import tqdm
from dataset import SSCM_dataloader
from utils import metrics
from core.res_unet import ResUnet
from core.res_unet_plus import ResUnetPlusPlus
from core.res2net_v2 import SSCM_net, SSD_net
from utils.logger import MyWriter
import torch
from torch.utils.data import DataLoader, RandomSampler
import argparse
import os


def main(hp, num_epochs, resume, name):

    checkpoint_dir = "{}/{}".format(hp.checkpoints, name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    os.makedirs("{}/{}".format(hp.log, name), exist_ok=True)
    writer = MyWriter("{}/{}".format(hp.log, name))
    # get model

    if hp.PLUS:
        model = SSD_net()
    else:
        #model = ResUnet(4)
        model = SSCM_net()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model.to(device)
    # set up binary cross entropy and dice loss
    criterion_SS = metrics.SS_Loss()
    criterion_CM = metrics.COMAP_Loss()


    # optimizer
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=True)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)

    # decay LR
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # starting params
    best_loss = 999
    start_epoch = 0
    # optionally resume from a checkpoint
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)

            start_epoch = checkpoint["epoch"]
            step = checkpoint["step"]

            best_loss = checkpoint["best_loss"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # get data
    if hp.PLUS:
        training_set = SSCM_dataloader.SSCM_dataset(hp.gn_root, hp.camera, rgb_only=True)
        validation_set = SSCM_dataloader.SSCM_dataset(hp.gn_root, hp.camera, split='valid', rgb_only=True)
    else:
        training_set = SSCM_dataloader.SSCM_dataset(hp.gn_root, hp.camera, rgb_only=False)
        validation_set = SSCM_dataloader.SSCM_dataset(hp.gn_root, hp.camera, split='valid', rgb_only=False)
    object_list = training_set.get_onject_list()

    # creating loaders
    train_sampler = RandomSampler(training_set)
    train_batch_sampler = SSCM_dataloader.ImageSizeBatchSampler(train_sampler, hp.batch_size, False, cfg=hp)
    train_dataloader = DataLoader(training_set, batch_sampler=train_batch_sampler, num_workers=hp.batch_size)

    val_dataloader = DataLoader(validation_set, batch_size=1, num_workers=0, shuffle=False)

    if not resume:
        step = 0
    for epoch in range(start_epoch, num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)


        # run training and validation
        # logging accuracy and loss
        #train_acc = metrics.MetricTracker()
        train_loss = metrics.MetricTracker()
        loss_ss = metrics.MetricTracker()
        loss_cm = metrics.MetricTracker()
        # iterate over data

        loader = tqdm(train_dataloader, desc="training")
        for idx, data in enumerate(loader):

            # get the inputs and wrap in Variable
            inputs = data["rgbd"].to(device)
            # print('input', inputs.shape)
            labels = data["gt"].to(device)
            # print('label', labels[:,6,:,:].shape)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # prob_map = model(inputs) # last activation was a sigmoid
            # outputs = (prob_map > 0.3).float()
            outputs = model(inputs)
            # print('output', torch.unsqueeze(outputs[:,6,:,:], 1).shape)
            # outputs = torch.nn.functional.sigmoid(outputs)

            SS = outputs[:, :len(object_list), ...]
            CM = outputs[:, len(object_list):, ...]
            SS_L = labels[:, 0, ...]
            # CM_L = labels[:, 1:, ...]
            l_ss = criterion_SS(SS, SS_L)
            l_cm = criterion_CM(CM, labels, use_mask=True)

            loss = l_ss + l_cm

            # backward
            loss.backward()
            optimizer.step()

            # train_acc.update(metrics.dice_coeff(outputs, labels), outputs.size(0))
            train_loss.update(loss.data.item(), outputs.size(0))
            loss_ss.update(l_ss.data.item(), outputs.size(0))
            loss_cm.update(l_cm.data.item(), outputs.size(0))

            # tensorboard logging
            if step % hp.logging_step == 0:
                writer.log_training(train_loss.avg, step)
                loader.set_description(
                    "Training Loss: {:.4f}, L_ss: {:.4f}, L_cm: {:.4f}".format(
                        train_loss.avg, loss_ss.avg, loss_cm.avg
                    )
                )
            step += 1

        # Validatiuon
        if epoch % hp.validation_interval == 0:
            valid_metrics = validation(
                val_dataloader, model, [criterion_SS, criterion_CM], writer, step, device
            )
            save_path = os.path.join(
                checkpoint_dir, "%s_checkpoint_%04d.pt" % (name, epoch+1)
            )
            # store best loss and save a model checkpoint
            best_loss = min(valid_metrics["valid_loss"], best_loss)
            torch.save(
                {
                    "step": step,
                    "epoch": epoch,
                    "arch": "ResUnet",
                    "state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                },
                save_path,
            )
            print("Saved checkpoint to: %s" % save_path)
        # step the learning rate scheduler
        lr_scheduler.step()


def validation(valid_loader, model, criterions, logger, step, device):

    # logging accuracy and loss
    #valid_acc = metrics.MetricTracker()
    valid_loss = metrics.MetricTracker()
    object_list = SSCM_dataloader.object_list

    # switch to evaluate mode
    model.eval()

    # Iterate over data.
    for idx, data in enumerate(tqdm(valid_loader, desc="validation")):

        # get the inputs and wrap in Variable
        inputs = data["rgbd"].to(device)
        labels = data["gt"].to(device)

        # forward
        # prob_map = model(inputs) # last activation was a sigmoid
        # outputs = (prob_map > 0.3).float()
        #print("v_shape: ", inputs.shape)
        outputs = model(inputs)
        # outputs = torch.nn.functional.sigmoid(outputs)
        SS = outputs[:, :len(object_list), ...]
        CM = outputs[:, len(object_list):, ...]
        SS_L = labels[:, 0, ...]
        # CM_L = labels[:, 1:, ...]

        loss = criterions[0](SS, SS_L) + criterions[1](CM, labels, use_mask=True)
        #print("Loss_SS: ", criterions[0](SS, SS_L), "   Loss_CM: ", criterions[1](CM, CM_L))

        #valid_acc.update(metrics.dice_coeff(outputs, labels), outputs.size(0))
        valid_loss.update(loss.data.item(), outputs.size(0))
        if idx == 0:
            logger.log_images(inputs.cpu(), labels.cpu(), outputs.cpu(), step)
    logger.log_validation(valid_loss.avg, step)

    print("Validation Loss: {:.4f}".format(valid_loss.avg))
    model.train()
    return {"valid_loss": valid_loss.avg}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic segmentation and 6DCM Prediction")
    # parser.add_argument("-c", "--config", type=str, required=True, help="yaml file for configuration")
    parser.add_argument("-c", "--config", type=str, default='configs/default.yaml', help="yaml file for configuration")
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument("--name", default="SSCM", type=str, help="Experiment name")

    args = parser.parse_args()

    hp = HParam(args.config)
    with open(args.config, "r") as f:
        hp_str = "".join(f.readlines())

    main(hp, num_epochs=args.epochs, resume=args.resume, name=args.name)
