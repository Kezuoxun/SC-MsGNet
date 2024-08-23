# adapted from https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard
from tensorboardX import SummaryWriter
from PIL import Image
import numpy as np


class MyWriter(SummaryWriter):
    def __init__(self, logdir):
        super(MyWriter, self).__init__(logdir)

    def log_training(self, smooth_l1_loss, step):
        self.add_scalar("training/smooth_l1_loss", smooth_l1_loss, step)

    def log_validation(self, smooth_l1_loss, step):
        self.add_scalar("validation/smooth_l1_loss", smooth_l1_loss, step)

    def log_images(self, rgbd, target, prediction, step):
        if len(rgbd.shape)>3:
            rgbd = rgbd.squeeze(0)
        if len(target.shape)>3:
            target = target.squeeze(0)
        if len(prediction.shape)>3:
            prediction = prediction.squeeze(0)
        self.add_image("rgb", rgbd[:2,:,:], step)
        #self.add_image("comap_f", target[:3,:,:], step)
        #self.add_image("prediction_f", prediction[:3,:,:], step)
        #self.add_image("comap_r", target[3:,:,:], step)
        #self.add_image("prediction_r", prediction[3:,:,:], step)


class LogWriter(SummaryWriter):
    def __init__(self, logdir):
        super(LogWriter, self).__init__(logdir)

    def log_scaler(self, key, value, step, prefix="Training", helper_func=None):
        if helper_func:
            value = helper_func(value)
        self.add_scalar("{}/{}".format(prefix, key), value, step)

    def log_image(self, key, value, step, prefix="Training", helper_func=None):
        if helper_func:
            value = helper_func(value)
        self.add_image("{}/{}".format(prefix, key), value, step)
