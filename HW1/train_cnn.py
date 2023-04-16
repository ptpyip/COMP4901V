import torch
from torchvision import transforms
from torchvision.transforms import RandomHorizontalFlip

from models import CNNClassifier, SoftmaxCrossEntropyLoss
from dataloders import load_data
import tools

DATASET_DIR = "../datasets/Vehicles"       # train or val
SAVE_DIR = "cnn_classifier"

def predict(scores: torch.Tensor):
    return scores.argmax(dim=-1)

# data augmentation
def getTransforms(is_train: bool = False):
    trans = [
        transforms.Resize((224, 224))
    ]
    
    if is_train:
        # Nomalization is done implicitly 
        # trans.append(RandomAffine( 
        #     degrees = 10, translate = (0.1, 0.1), 
        #     scale = (0.9, 1.9) ,shear = 10
        # ))
        trans.append(RandomHorizontalFlip(p=0.5))
        
    
    return transforms.Compose(trans)

def train(args):
    """
    Your code here
    """    
    args["dataset_dir"] = "../datasets/Vehicles"
    tools.train(
        model=CNNClassifier(), 
        load_data=load_data,
        loss_fn=SoftmaxCrossEntropyLoss(),
        getTransforms=getTransforms,
        predict=predict,
        args=args
    )


if __name__ == '__main__':
    args = {
        "cuda": 0,
        "save_dir": "./ckpts/cnn_classifier",
        "lr": 1e-4,
        "momentum": 0.9,
        "wd": 1e-3,
        "epochs": 30,
        "log_interval": 2,
    }
    train(args)
