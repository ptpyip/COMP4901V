import os
import numpy as np
from PIL import Image
import matplotlib as mpl

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

save_dir = "./results"

SEGMENTATION_COLORS = [  # [  0,   0,   0],
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [0, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
]


def visualize(model, metrics, predict, dataloader, cuda=None, multi_task=False):
    quant_results = {
        "best_acc": {
            "i": -1,
            "acc": [0] ,
            "preds": [],
            "data": []
        },
        "lowest_acc": {
            "i": -1,
            "acc": [100] ,
            "preds": [],
            "data": []
        },
        "best_rel": {
            "i": -1,
            "rel": [0] ,
            "preds": [],
            "data": []
        },
        "lowest_rel": {
            "i": -1,
            "rel":  [100] ,
            "preds": [],
            "data": []
        },
        "most_label": {
            "i": -1,
            "num_of_label": 0. ,
            "preds": [],
            "data": []
        },
        "least_label": {
            "i": -1,
            "num_of_label": 20 ,
            "preds": [],
            "data": []
        },
    }
    with torch.no_grad():
        model.eval()
        
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
            accuracy, avg_accuracy, mIoU = metrics.matrix.global_accuracy, metrics.matrix.average_accuracy, metrics.matrix.iou
            rel, a1, a2, a3 = metrics.depth_error.compute_errors
            metrics.clear()
            
            num_of_label = labels[0].unique().shape[0]
            result = {
                    "i": i,
                    "acc": [accuracy, avg_accuracy, mIoU] ,
                    "rel": [rel, a1, a2, a3] ,
                    "num_of_label": num_of_label ,
                    "preds": preds,
                    "data": [inputs, labels]
                }
            if accuracy > quant_results["best_acc"]["acc"][0]:
                # store best accuracy 
                quant_results["best_acc"] = result
                
            elif accuracy < quant_results["lowest_acc"]["acc"][0]:
                # store lowest accuracy 
                quant_results["lowest_acc"] = result
            elif  rel > quant_results["best_rel"]["rel"][0]:
                # store best rel        
                quant_results["best_rel"] = result
            elif  rel < quant_results["lowest_rel"]["rel"][0]:
                # store lowest rel   
                quant_results["lowest_rel"] = result
            elif  num_of_label > quant_results["most_label"]["num_of_label"]:  
                # more label
                quant_results["most_label"] = result
            elif  num_of_label < quant_results["least_label"]["num_of_label"]:  
                # fewest label
                quant_results["least_label"] = result
        
        for key, result in quant_results.items():
            seg_map_preds, depth_esteiates = result["preds"]
            img, (seg_map_tgt, detph_tgt) = result["data"]
            
            vis = DenseVisualization(img, (depth_esteiates, detph_tgt), (seg_map_preds, seg_map_tgt))
            vis.save(key)
            
            accuracy, avg_accuracy, mIoU = result["acc"]
            rel, a1, a2, a3 = result["rel"]
            i = result["i"]
            print(f"{key}: {i}")
            printResults(accuracy, avg_accuracy, mIoU, rel, a1, a2, a3)
            
            num_of_label = result["num_of_label"]
            print(f"Number of label: {num_of_label}")
    # return metrics
    
def printResults(accuracy, avg_accuracy, mIoU, rel, a1, a2, a3):
    print(f"Segmentation accuracy: {accuracy}")
    print(f"Segmentation avg accuracy: {avg_accuracy}")
    print(f"Segmentation mIOU: {mIoU}")

    
    print(f"Detph mean relative error: {rel}")
    print(f"Detph accuracy < t_1: {a1}")
    print(f"Detph accuracy < t_2: {a2}")
    print(f"Detph accuracy < t_3: {a3}")
    
            
class DenseVisualization():
    def __init__(self, img, depth, segmentation):
        self.img = img.reshape(*img.shape[1:]).permute(1, 2, 0).cpu().numpy() * 256 
        # self.depth = depth
        # self.segmentation = segmentation
        self.seg_maps = [
            self.colorize_segmap(segmentation[0].reshape(*img.shape[2:])),
            self.colorize_segmap(segmentation[1].reshape(*img.shape[2:])),
        ]
        self.depth_maps = [
            self.colorize_depth(depth[0].reshape(*img.shape[2:])),
            self.colorize_depth(depth[1].reshape(*img.shape[2:])),               
        ]
        # print(self.img.shape)
        # print(segmentation[0].shape)
        # print(segmentation[1].shape)
        # print(depth[0].shape)
        # print(depth[1].shape)
        
    
    def save(self, name):
        """
        Your code here
        Hint: you can visualize your model predictions and save them into images. 
        """
        # sample 6 img
        if not os.path.isdir(f"{save_dir}/"):
            os.mkdir(f"{save_dir}")
        if not os.path.isdir(f"{save_dir}/{name}"):
            os.mkdir(f"{save_dir}/{name}")
        
        im = Image.fromarray(self.img.astype(np.uint8))
        im.save(f"{save_dir}/{name}img.jpeg")
        for i in range(2):
            seg = Image.fromarray( self.seg_maps[i].astype(np.uint8))
            seg.save(f"{save_dir}/{name}/seg_{i}.jpeg")
            depth = Image.fromarray( self.depth_maps[i].astype(np.uint8))
            depth.save(f"{save_dir}/{name}/depth_{i}.jpeg")
     
    def colorize_segmap(self, seg_map):
        '''
            source:https://github.com/shashankag14/Cityscapes-Segmentation/blob/master/Cityscapes_modified_R2UNET.ipynb
        '''
        seg_map = seg_map.cpu().numpy()
        rgb = np.zeros((*seg_map.shape, 3))        
        for c in range(19):
            rgb[:, :][seg_map == c] = SEGMENTATION_COLORS[c]

        return rgb
    
    def colorize_depth(self, depth_esteiates):
        depth_esteiates = depth_esteiates.cpu().numpy()
        depth_esteiates[depth_esteiates<0] = 0
        
        d_min = depth_esteiates.min()
        d_max =  np.percentile(depth_esteiates, 95)
        # depth_normalized = (depth_esteiates - d_min) / (d_max - d_min)
        
        # return 255 * depthCM(depth_normalized)[:, :, :3]
        normalizer = mpl.colors.Normalize(vmin=d_min, vmax=d_max)
        mapper = mpl.cm.ScalarMappable(norm=normalizer, cmap='magma')
        return mapper.to_rgba(depth_esteiates)[:, :, :3] * 255

# def label_to_pil_image(lbl):
#     """
#     Creates a PIL pallet Image from a pytorch tensor of labels
#     """
#     if not(isinstance(lbl, torch.Tensor) or isinstance(lbl, np.ndarray)):
#         raise TypeError('lbl should be Tensor or ndarray. Got {}.'.format(type(lbl)))
#     elif isinstance(lbl, torch.Tensor):
#         if lbl.ndimension() != 2:
#             raise ValueError('lbl should be 2 dimensional. Got {} dimensions.'.format(lbl.ndimension()))
#         lbl = lbl.numpy()
#     elif isinstance(lbl, np.ndarray):
#         if lbl.ndim != 2:
#             raise ValueError('lbl should be 2 dimensional. Got {} dimensions.'.format(lbl.ndim))

#     im = Image.fromarray(lbl.astype(np.uint8), mode='P')
#     # im.putpalette([0xee, 0xee, 0xec, 0xfc, 0xaf, 0x3e, 0x2e, 0x34, 0x36, 0x20, 0x4a, 0x87, 0xa4, 0x0, 0x0] + [0] * 753)
#     im.putpalette([int(255/19) * (i) for i in range(19)])
#     return im
