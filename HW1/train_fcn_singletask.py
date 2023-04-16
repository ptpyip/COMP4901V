import torch
import torch.nn as nn
import numpy as np

from models import FCN_ST, SoftmaxCrossEntropyLoss
import tools.dense_transforms as dense_transforms
import torch.utils.tensorboard as tb

import tools
import tools.dense_transforms as T
from dataloders import load_dense_data

CLASS_WEIGHTS = [
    3.29, 21.9, 4.68, 121.32, 266.84,
    117.6, 1022.23, 205.68, 6.13, 118.81,
    35.17, 168.36, 460.62, 15.53, 272.62,
    501.94, 3536.12, 2287.91, 140.32
]

class PixelwiseCrossEntropyLoss(nn.Module):
    def __init__(self, classes_weights, cuda=None):
        super(PixelwiseCrossEntropyLoss, self).__init__()
        
        self.classes_weights = torch.tensor(classes_weights)
        if torch.cuda.is_available() and cuda!=None : 
            self.classes_weights.cuda(cuda)
            
        self.CELoss = nn.CrossEntropyLoss(weight=self.classes_weights, ignore_index=255)
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Your code here
        @inputs: torch.Tensor((B,C,H,W)), C is the number of classes for segmentation.
        @targets: torch.Tensor((B,1,H,W))
        """
        
        # B,C,H,W = inputs.shape
        
        # loss = 0.
        # for c in range(C):
        #     loss += self.CELoss(inputs[:, c, :, :], targets) * self.classes_weights[c]
        
        return self.CELoss(inputs, targets)          
        
# data augmentation
def getTransforms(is_train: bool = False):
    trans = []
    
    if is_train:
        # Nomalization is done implicitly 
        trans.append(T.RandomHorizontalFlip(flip_prob=0.5))
    
    return T.Compose(trans)   

def predict(scores_map: torch.Tensor):
    """
    Your code here
    @inputs: torch.Tensor((B,C,H,W)), C is the number of classes for segmentation.
    """
    return scores_map.argmax(dim=1)

def train(args):
    """
    Your code here
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    """
    args["dataset_dir"] = "../datasets/DenseCityscapesDataset"
    tools.train(
        model=FCN_ST(), 
        load_data=load_dense_data,
        loss_fn=PixelwiseCrossEntropyLoss(CLASS_WEIGHTS, args.get("cuda")),
        getTransforms=getTransforms,
        predict=predict,
        args=args
    )

    return

def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    args = {
        "cuda": 2,
        "save_dir": "./ckpts/fcn_seg",
        "lr": 1e-3,
        "momentum": 0.9,
        "wd": 1e-3,
        "epochs": 750,
        "log_interval": 2,
    }
    train(args)
