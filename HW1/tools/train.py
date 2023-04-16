import os
from datetime import datetime

import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import RandomAffine

import torch.utils.tensorboard as tb

from tools.engine import Engine, Logger

DATASET_DIR = "../datasets/Vehicles"       # train or val
SAVE_DIR = "./"


def train(model, load_data, loss_fn, getTransforms, predict, args):
    """
    Your code here
    """    
    
    ## Initailize model + load ckpt
    start_epoch = 0
    if args.get("ckpt_path"):
        ckpt = torch.load(args.get("ckpt_path"), map_location='cpu')
        model.load_state_dict(ckpt["state_dict"])
        start_epoch = ckpt["epoch"]
        
    if torch.cuda.is_available(): 
        model.cuda(args.get("cuda"))
        loss_fn.cuda(args.get("cuda"))
        
    print(f"Using GPU: {next(model.parameters()).is_cuda}")
    
    ## Initailize datloaders 
    dataset_dir = args.get("dataset_dir") if args.get("dataset_dir") else DATASET_DIR
    train_datloader = load_data(f"{dataset_dir}/train/", transforms=getTransforms(is_train=True), depth_reqd=args.get("multi_task"))
    val_datloader = load_data(f"{dataset_dir}/val/", transforms=getTransforms(), depth_reqd=args.get("multi_task"))
        
    ## Initailize loggers     
    save_dir = args.get("save_dir") if args.get("save_dir") else SAVE_DIR
    logger = Logger(save_dir, predict, model.C, multi_task=args.get("multi_task"))
        
    ## Initailize engine 
    engine = Engine(model, 
        loss_fn=loss_fn,
        logger=logger,
        args=args,
        log_interval=args.get("log_interval"),
        cuda=args.get("cuda")
    )
    
    ## training epochs 
    print("start training")
    best_vloss = 1_000_000
    for epoch in range(start_epoch, args.get("epochs")):
        print(f'EPOCH {epoch}')
        
        ### train for one epoch
        model.train(True)
        train_loss = engine.run(train_datloader, epoch, isTraining=True)
        train_log_msg = logger.logMetrics("Training", epoch+1, miou=args.get("mIOU"))
        
        ### validation
        with torch.no_grad():
            model.eval()
            val_loss = engine.run(val_datloader, epoch)
            val_log_msg = logger.logMetrics("Validation", epoch+1, miou=args.get("mIOU"))
           
        ### log accuracy       
        print(train_log_msg)  
        print(val_log_msg)                
        
        ### write to TB                
        logger.writer.add_scalars('Training vs. Validation Loss',{ 
            'Training' : train_loss, 
            'Validation' : val_loss 
        }, epoch + 1)
        logger.writer.flush()
        
        ### Save checkpoint
        engine.saveModel(epoch, val_loss, save_dir)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
