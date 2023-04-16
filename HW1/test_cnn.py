import os
from datetime import datetime

import torch

import torch.utils.tensorboard as tb

from models import CNNClassifier
from dataloders import load_data
from train_cnn import getTransforms, predict
from tools.eval import eval, ConfusionMatrix

def test(args):
    model = CNNClassifier()
    if args.get("ckpt_path"):
        ckpt = torch.load(args.get("ckpt_path"), map_location='cpu')
        model.load_state_dict(ckpt["state_dict"])
        
    if torch.cuda.is_available(): 
        model.cuda(args.get("cuda"))
        
    print(f"Using GPU: {next(model.parameters()).is_cuda}")
    
    ## Initailize datloaders 
    dataset_dir = args.get("dataset_dir") 
    test_datloader = load_data(dataset_dir, transforms=getTransforms())
               
    metrics = eval(model, ConfusionMatrix(model.C), predict, test_datloader, args.get("cuda"))
    print(f"Accuracy: {metrics.global_accuracy}")
    print(f"Avg accuracy: {metrics.average_accuracy}")

if __name__ == '__main__':
    args = {
        "cuda": 0,
        "save_dir": "./ckpts//cnn_classifier",
        "dataset_dir": "../datasets/Vehicles/val",
        "ckpt_path": "./ckpts/cnn_classifier/epoch_30_0331_122233.pth"
    }
    test(args)
