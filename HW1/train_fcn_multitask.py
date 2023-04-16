import torch
import torch.nn as nn
import numpy as np


from models import FCN_MT
from tools.eval import MultiTaskMetrics
import tools.dense_transforms as T
import torch.utils.tensorboard as tb

import tools
from dataloders import load_dense_data
from train_fcn_singletask import PixelwiseCrossEntropyLoss, CLASS_WEIGHTS

def predict(inputs: torch.Tensor):
    """
    Your code here
    @inputs: torch.Tensor((B,C,H,W)), C is the number of classes for segmentation.
    """
    seg_map_sources, depth_esteiates = inputs
    seg_map = seg_map_sources.argmax(dim=1)
    
    return seg_map, depth_esteiates

class CustomDenseLoss(nn.Module):
    def __init__(self, classes_weights, cuda=None):
        super(CustomDenseLoss, self).__init__()
            
        self.L1_loss = nn.L1Loss()
        self.CE_Loss = PixelwiseCrossEntropyLoss(classes_weights, cuda)
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        maks out invalid labels (-ve values) perfor pass to L1 loss
        @inputs: torch.Tensor((B,1,H,W)), C is the number of classes for segmentation.
        @targets: torch.Tensor((B,1,H,W))
        """
        seg_map_sources, depth_esteiates = inputs
        # print(depth_esteiates[0, :, 0])
        seg_map_tgt, detph_tgt = targets
        
        seg_loss = self.CE_Loss(seg_map_sources, seg_map_tgt)
        
        mask = detph_tgt > 0
        # depth_esteiates[detph_tgt < 0] = 0
        # detph_tgt[detph_tgt < 0] = 0
        # print(depth_esteiates[mask].isfinite().unique())
        # print(detph_tgt[mask].isfinite().unique())
        
        depth_loss = self.L1_loss(depth_esteiates[mask], detph_tgt[mask])
        # depth_loss = torch.abs(detph_tgt[mask]-depth_esteiates[mask]).median()
        # print(depth_loss)
        
        return seg_loss + depth_loss
  
# data augmentation
def getTransforms(is_train: bool = False):
    trans = []
    
    if is_train:
        # Nomalization is done implicitly 
        trans.append(T.RandomHorizontalFlip3(flip_prob=0.5))
        # trans.append(T.RandomAffine3( 
        #     degrees = 10, translate = (0.1, 0.1), 
        #     scale = (0.9, 1.9) ,shear = 10
        # ))
    
    return T.Compose3(trans)                 

def train(args):

    """
    Your code here
    Hint: validation during training: use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: Use dense_transforms for data augmentation. If you found a good data augmentation parameters for the CNN, use them here too. 
    Hint: Use the log function below to debug and visualize your model
    """
    args["dataset_dir"] = "../datasets/DenseCityscapesDataset"
    args["multi_task"] = True
    
    model = FCN_MT()
    fcn_seg = torch.load(args["fcn_st path"], map_location='cpu')
    model.load_state_dict(fcn_seg["state_dict"], strict=False)
    
    tools.train(
        model=model, 
        load_data=load_dense_data,
        loss_fn=CustomDenseLoss(CLASS_WEIGHTS, args.get("cuda")),
        getTransforms=getTransforms,
        predict=predict,
        args=args
    )


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    # """
    # logger.add_image('image', imgs[0], global_step)
    # logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
    #                                          convert('RGB')), global_step, dataformats='HWC')
    # logger.add_image('prediction', np.array(dense_transforms.
    #                                               label_to_pil_image(logits[0].argmax(dim=0).cpu()).
    #                                               convert('RGB')), global_step, dataformats='HWC')
    ...

if __name__ == '__main__':
    args = {
        "cuda":5,
        "save_dir": "./ckpts/fcn_mt",
        "lr": 1e-4,
        "momentum": 0.9,
        "wd": 1e-3,
        "epochs": 100,
        "log_interval": 2,
        "fcn_st path": "./ckpts/fcn_seg/epoch_750_0402_193945.pth"
    }
    train(args)
