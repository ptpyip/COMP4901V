import os
from datetime import datetime

import torch
import torchvision
import torch.utils.tensorboard as tb
from torch.optim import lr_scheduler 

from tools.eval import ConfusionMatrix, MultiTaskMetrics

class Logger():
    def __init__(self, save_dir, predict, num_class, multi_task=False) -> None:
        self.timestamp = datetime.now().strftime('%m%d_%H%M%S')
        self.writer = tb.SummaryWriter(f'{save_dir}/runs/train_{self.timestamp}')
        self.predict = predict
        self.multi_task = multi_task
        self.metrics = ConfusionMatrix(size=num_class) if not multi_task else MultiTaskMetrics(size=num_class) 
                
    def addMetrics(self, scores, labels):      
        preds = self.predict(scores)
        self.metrics.add(preds, labels)
        
    def logMetrics(self, name, epoch_num, miou=None):
        log_msg = self.logAcc(name, epoch_num)
        if miou:
            log_msg += "\n" 
            log_msg += self.logIOU(name, epoch_num)
            
        if self.multi_task:
            log_msg += "\n" 
            log_msg += self.logRelError(name, epoch_num)
            
        self.metrics.clear()
        return log_msg
        
    def logAcc(self, name, epoch_num):
        log_msg = f"{name} accuracy: {self.metrics.global_accuracy}"
        self.writer.add_scalar(f"{name} accuracy:", self.metrics.global_accuracy, epoch_num)
        return log_msg
    
    def logIOU(self, name, epoch_num):
        log_msg = f"{name} mIOU: {self.metrics.iou}"
        self.writer.add_scalar(f"{name} mIOU:", self.metrics.iou, epoch_num)
        return log_msg
    
    def logRelError(self, name, epoch_num):
        rel_error = self.metrics.compute_errors[0]
        log_msg = f"{name} abs rel error: {rel_error}"
        self.writer.add_scalar(f"{name} Relative error:", rel_error, epoch_num)
        return log_msg
        
class Engine():
    '''A class wrapper for the training pipeline
    '''
    model: torch.nn.Module
    best_vloss = 1_000_000
    
    def __init__(self, model, loss_fn, logger, args, log_interval=10, cuda=None) -> None:
        self.model = model
        self.loss_fn = loss_fn 
        self.args = args
        self.log_interval = log_interval
        self.cuda = cuda
        
        self.optimizer = self.setOptimizer()
        if args.get("lr_decay"):
            self.scheduler = self.setScheduler()
         
        self.multi_task = args.get("multi_task")
        self.logger = logger
                    
    def run(self, dataloader, epoch_num, isTraining=False):
        if isTraining: 
            self.optimizer.zero_grad()
                    
        running_loss = 0.
        last_loss = 0.    
        
        ### iterate all data batch      
        for i, data in enumerate(dataloader):
            inputs, labels = data
            if torch.cuda.is_available(): 
                inputs = inputs.cuda(self.cuda)
                if self.multi_task: 
                    labels = [label.cuda(self.cuda) for label in labels]
                else:
                    labels = labels.cuda(self.cuda)
            
            scores = self.model(inputs)
            loss = self.loss_fn(scores, labels)
            running_loss += loss.item()
                        
            if isTraining:
                loss.backward()
                self.optimizer.step()
                if self.args.get("lr_mode"): self.scheduler.step()
                
                if (i % self.log_interval == self.log_interval-1):
                    last_loss = running_loss /  self.log_interval # loss per batch
                    print(f'\tbatch {i+1} loss: {last_loss}')
                    
                    tb_x = epoch_num * len(dataloader) + i + 1
                    self.logger.writer.add_scalar('Loss/train', last_loss, tb_x)
                    
                    running_loss = 0.
             
            # if self.multi_task: 
            #     scores, _ = scores       
            #     labels, _ = labels
            self.logger.addMetrics(scores, labels)
                
        if not isTraining : 
            last_loss = running_loss / (i + 1)
            print(f'\tval loss: {last_loss}')
            self.logger.writer.add_scalar('Loss/Val', last_loss, epoch_num+1)
            
        elif self.args.get("lr_decay"):
            self.scheduler.step()
            
        return last_loss        
    
    def setOptimizer(self):
        learning_rate = self.args.get("lr")
        momentum = self.args.get("momentum")
        weight_decay = self.args.get("wd")
        
        # for possible fintuning optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.SGD(params, learning_rate, momentum, 
                            weight_decay=weight_decay)
    
    def setScheduler(self):
        scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.args.get("setp_size"), gamma=self.args.get("lr_decay"), verbose=True)
        return scheduler
    
    def setLogger(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return tb.SummaryWriter(f'runs/{self.cfg.name}_{timestamp}')
    
    def saveModel(self, epoch, val_loss, save_dir):   
        if not os.path.exists(f"{save_dir}/ckpts"): 
            os.mkdir(f"{save_dir}/ckpts")
            
        torch.save({
            "epoch": epoch+1,
            "args": self.args,
            "state_dict": self.model.state_dict()
        }, f'{save_dir}/ckpts/last_{self.logger.timestamp}.pth')
        
        if (epoch+1)% 10 == 0:
            torch.save({
                "epoch": epoch+1,
                "args": self.args,
                "state_dict": self.model.state_dict()
            }, f'{save_dir}/ckpts/epoch_{epoch+1}_{self.logger.timestamp}.pth')
            
        ### Track best performance, and save the model's state
        if val_loss < self.best_vloss:
            self.best_vloss = val_loss
            model_path = f'{save_dir}/ckpts/best_{self.logger.timestamp}.pth'
            torch.save({
                "epoch": epoch+1,
                "args": self.args,
                "state_dict": self.model.state_dict()
            }, model_path)

    
    