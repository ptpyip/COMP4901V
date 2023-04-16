import torch
import numpy as np

def eval(model, metrics, predict, dataloader, cuda=None, multi_task=False):
    with torch.no_grad():
        model.eval()
        
        num_correct = 0
        for i, data in enumerate(dataloader):
            inputs, labels = data
            if torch.cuda.is_available(): 
                inputs = inputs.cuda(cuda)
                if multi_task: 
                    labels = [label.cuda(cuda) for label in labels]
                else: 
                    labels = labels.cuda(cuda)
            
            scores = model(inputs)
            preds = predict(scores)
            metrics.add(preds, labels)
            
            # num_correct += torch.sum(
            #     (preds == labels).type(torch.float)      # a tensor of 0/1
            # ).item()
            
        # accuracy =  num_correct / (i+1)
        return metrics
    
def _one_hot(x, n):
    return (x.view(-1, 1) == torch.arange(n, dtype=x.dtype, device=x.device)).int()

class ConfusionMatrix(object):
    def _make(self, preds, labels):
        label_range = torch.arange(self.size, device=preds.device)[None, :]
        preds_one_hot, labels_one_hot = _one_hot(preds, self.size), _one_hot(labels, self.size)
        return (labels_one_hot[:, :, None] * preds_one_hot[:, None, :]).sum(dim=0).detach()

    def __init__(self, size=5):
        """
        This class builds and updates a confusion matrix.
        :param size: the number of classes to consider
        """
        self.matrix = torch.zeros(size, size)
        self.size = size

    def add(self, preds, labels):
        """
        Updates the confusion matrix using predictions `preds` (e.g. logit.argmax(1)) and ground truth `labels`
        """
        self.matrix = self.matrix.to(preds.device)
        self.matrix += self._make(preds, labels).float()
        
    def clear(self):
        self.matrix = torch.zeros(self.size, self.size)

    @property
    def class_iou(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(0) + self.matrix.sum(1) - true_pos + 1e-5)

    @property
    def iou(self):
        return self.class_iou.mean()

    @property
    def global_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos.sum() / (self.matrix.sum() + 1e-5)

    @property
    def class_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(1) + 1e-5)

    @property
    def average_accuracy(self):
        return self.class_accuracy.mean()

    @property
    def per_class(self):
        return self.matrix / (self.matrix.sum(1, keepdims=True) + 1e-5)

class DepthError(object):
    def __init__(self, gt, pred):
        """
        This class builds and updates a confusion matrix.
        :param size: the number of classes to consider
        """
        self.gt = gt
        self.pred = pred

    @property
    def compute_errors(self):
        """Computation of error metrics between predicted and ground truth depths
        """
        
        # mask = (self.gt > 0) & (self.pred>0)
        mask = (self.gt > 0)
        masked_detph_tgt = self.gt[mask]
        masked_depth_esteiates = self.pred[mask]
        
        # gt = masked_detph_tgt.cpu().detach().numpy()
        gt = masked_detph_tgt.cpu().detach().numpy()
        pred = masked_depth_esteiates.cpu().detach().numpy()
        
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = thresh[thresh < 1.25].mean()
        a2 = thresh[thresh < 1.25**2].mean()
        a3 = thresh[thresh < 1.25**3].mean()
        a = (np.abs(gt - pred) / gt)
        
        abs_rel = np.mean(np.abs(gt - pred) / gt) 
        # abs_rel = np.abs(gt - pred) / gt
        # mean_abs_rel = np.mean(np.nan_to_num(abs_rel))

        return abs_rel, a1, a2, a3
    
    def add(self,  gt, pred):
        self.gt = torch.vstack((self.gt, gt))
        self.pred = torch.vstack((self.pred, pred))
        
def getDepthError(pred, gt):
    
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    return abs_rel, a1, a2, a3


class MultiTaskMetrics(object):
    
    @staticmethod
    def getDepthError(pred, gt):
        
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25     ).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        abs_rel = np.mean(np.abs(gt - pred) / gt)
            
        return abs_rel, a1, a2, a3
    
    def __init__(self, size) -> None:
        self.matrix = ConfusionMatrix(size=size)
        self.depth_error = None
            
    def add(self, preds, labels):
        seg_map_preds, depth_esteiates = preds 
        seg_map_tgt, detph_tgt = labels
        
        self.matrix.add(seg_map_preds, seg_map_tgt)
        
        if (self.depth_error == None):
            self.depth_error = DepthError(detph_tgt, depth_esteiates)
        else: 
            self.depth_error.add(detph_tgt, depth_esteiates)
    
    def clear(self):
        self.matrix.clear()
        self.depth_error = None
    
    @property
    def class_iou(self):
        return self.matrix.class_iou

    @property
    def iou(self):
        return self.matrix.iou

    @property
    def global_accuracy(self):
        return self.matrix.global_accuracy

    @property
    def class_accuracy(self):
        return self.matrix.class_accuracy

    @property
    def average_accuracy(self):
        return self.matrix.average_accuracy

    @property
    def per_class(self):
        return self.matrix.per_class
    
    @property
    def compute_errors(self):
        return self.depth_error.compute_errors