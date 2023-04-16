import os
from datetime import datetime

import torch

import torch.utils.tensorboard as tb

from models import FCN_MT
from dataloders import load_dense_data
from train_fcn_multitask import getTransforms, predict
from tools.eval import eval, MultiTaskMetrics

def test(args):
    from os import path
    """
    Your code here
    Hint: load the saved checkpoint of your model, and perform evaluation for both segmentation and depth estimation tasks
    Hint: use the ConfusionMatrix for you to calculate accuracy, mIoU for the segmentation task
    Hint: use DepthError for you to calculate rel, a1, a2, and a3 for the depth estimation task. 
    """
    model = FCN_MT()
    if args.get("ckpt_path"):
        ckpt = torch.load(args.get("ckpt_path"), map_location='cpu')
        model.load_state_dict(ckpt["state_dict"])
        
    if torch.cuda.is_available(): 
        model.cuda(args.get("cuda"))
        
    print(f"Using GPU: {next(model.parameters()).is_cuda}")
    
    ## Initailize datloaders 
    dataset_dir = args.get("dataset_dir") 
    test_datloader = load_dense_data(dataset_dir, transforms=getTransforms(), depth_reqd=True)
               
    metrics = eval(model, MultiTaskMetrics(model.C), predict, test_datloader, args.get("cuda"), multi_task=True)
    accuracy, avg_accuracy, mIoU = metrics.matrix.global_accuracy, metrics.matrix.average_accuracy, metrics.matrix.iou
    rel, a1, a2, a3 = metrics.depth_error.compute_errors
    
    print(f"Segmentation accuracy: {accuracy}")
    print(f"Segmentation avg accuracy: {avg_accuracy}")
    print(f"Segmentation mIOU: {mIoU}")

    
    print(f"Detph mean relative error: {rel}")
    print(f"Detph accuracy < t_1: {a1}")
    print(f"Detph accuracy < t_2: {a2}")
    print(f"Detph accuracy < t_3: {a3}")

    # return accuracy, mIoU, rel, a1, a2, a3


if __name__ == '__main__':
    args = {
        "cuda": 0,
        "save_dir": "./ckpts/fcn_mt",
        "dataset_dir": "../datasets/DenseCityscapesDataset/test",
        "ckpt_path": "./ckpts/fcn_mt/best_0405_011200.pth"
    }
    test(args)
