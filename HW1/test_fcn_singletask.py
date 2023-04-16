import os
from datetime import datetime

import torch

import torch.utils.tensorboard as tb

from models import FCN_ST
from dataloders import load_dense_data
from train_fcn_singletask import getTransforms, predict
from tools.eval import eval, ConfusionMatrix

def test(args):
    model = FCN_ST()
    if args.get("ckpt_path"):
        ckpt = torch.load(args.get("ckpt_path"), map_location='cpu')
        model.load_state_dict(ckpt["state_dict"])
        
    if torch.cuda.is_available(): 
        model.cuda(args.get("cuda"))
        
    print(f"Using GPU: {next(model.parameters()).is_cuda}")
    
    ## Initailize datloaders 
    dataset_dir = args.get("dataset_dir") 
    test_datloader = load_dense_data(dataset_dir, transforms=getTransforms())
               
    metrics = eval(model, ConfusionMatrix(model.C), predict, test_datloader, args.get("cuda"))
    print(f"Test accuracy: {metrics.global_accuracy}")
    print(f"Test avg accuracy: {metrics.average_accuracy}")
    print(f"Test mIOU: {metrics.iou}")
    
    return metrics.global_accuracy, metrics.iou

if __name__ == '__main__':
    args = {
        "cuda": 2,
        "save_dir": "./ckpts/fcn_seg",
        "dataset_dir": "../datasets/DenseCityscapesDataset/test",
        "ckpt_path": "./ckpts/fcn_seg/epoch_750_0402_193945.pth"
    }
    test(args)
