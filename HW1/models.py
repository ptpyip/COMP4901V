import torch

import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50, ResNet50_Weights

PRETRAINED_WEIGHTS = ResNet50_Weights.IMAGENET1K_V2
      
class ResNetEncoder(nn.Module):
    def __init__(self, skip_connect=False):
        super(ResNetEncoder, self).__init__()

        backbone = resnet50(weights=PRETRAINED_WEIGHTS)
        backbone_layers =  [layer for layer in backbone.children()]
        
        print("get backbone_layers")
        self.layers = nn.ModuleList([
            nn.Sequential(*backbone_layers[:4]),
            backbone_layers[4],                     # layer 1
            backbone_layers[5],                     # layer 2
            backbone_layers[6],                     # layer 3
            backbone_layers[7],                     # layer 4
        ])
        self.skip_connect = skip_connect
        
    def forward(self, x):
        outputs = []
        
        out = self.layers[0](x)
        # if self.skip_connect: outputs.append(out)
        
        out = self.layers[1](out)
        if self.skip_connect: outputs.append(out)
        
        out = self.layers[2](out)
        if self.skip_connect: outputs.append(out)
        
        out = self.layers[3](out)
        if self.skip_connect: outputs.append(out)
        
        out = self.layers[4](out)
        outputs.append(out)
        
        return outputs

class CNNClassifier(nn.Module):  
    def __init__(self):
        super().__init__()
        self.device = torch.cuda.device("cuda:0") if torch.cuda.is_available() else "cpu"
        self.C = 6      
        
        self.feature_encoder = ResNetEncoder()
        '''
            A CNN Feature Encoder using ResNet50 with pretrained ImageNet \n
            (with out AbgPool and FC layers at the end)
            - out: [N, 2048, 7, 7]
        '''
        print("get feature_encoder")
        
        self.classifier = nn.Sequential(
            nn.AvgPool2d(7, 1, 0),          # output: 2048x1x1
            nn.Flatten(),                   # output: Nx2048
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.C)
        )
        print("get classifier")
        

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        """
        # B, _, H, W = x.shape
        
        x_normalized = (x - x.mean()) / torch.sqrt(x.var())
        
        # self.feature_encoder.eval()
        # with torch.no_grad():
        feature_map = self.feature_encoder(x_normalized)[0]
        
        return self.classifier(feature_map)
    
    def __getBackboneLayers(self, pretrained_model):
        backbone = resnet50(weights=PRETRAINED_WEIGHTS)
        return [layer for layer in backbone.children()]

class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampling, self).__init__()
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
                
    def forward(self, x):
        return self.deconv(x)
   
class UpSamplingDecoder(nn.Module):
    def __init__(self):
        super(UpSamplingDecoder, self).__init__()
        
        self.deconv1 = UpSampling(2048, 1024)
        self.bn1     = nn.BatchNorm2d(1024)
        
        self.deconv2 = UpSampling(1024, 512)
        self.bn2     = nn.BatchNorm2d(512)
        
        self.deconv3 = UpSampling(512, 256)
        self.bn3     = nn.BatchNorm2d(256)
        
        self.deconv4 = UpSampling(256, 128)
        self.bn4     = nn.BatchNorm2d(128)
        
        self.deconv5 = UpSampling(128, 64)
        self.bn5     = nn.BatchNorm2d(64)
                
    def forward(self, feature_maps):
          
        out = self.deconv1(feature_maps[3])
        out = self.bn1(out + feature_maps[2])
        
        out = self.deconv2(out)
        out = self.bn2(out + feature_maps[1])
        
        out = self.deconv3(out)
        out = self.bn3(out + feature_maps[0])
        
        out = self.deconv4(out)
        out = self.bn4(out)
        # out = self.bn4(out + feature_maps[0])
        
        out = self.deconv5(out)
        out = self.bn5(out)
        
        return out    


class FCN_PredictionHead(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super(FCN_PredictionHead, self).__init__()
        self.prediction_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(32, out_channels, 1),
        )
    def forward(self, x):
        return self.prediction_head(x)
        
                                        
class FCN_ST(torch.nn.Module):
    def __init__(self):
        """
        Your code here.
        Hint: The Single-Task FCN need to output segmentation maps at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        super().__init__()
        self.C = 19      

        self.feature_encoder = ResNetEncoder(skip_connect=True)
        
        self.upsamling_decoder = UpSamplingDecoder()
        
        ## residual (input to decoder)
        self.residual = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1)
        )
                
        self.seg_head = FCN_PredictionHead(self.C)
        
        # ## decoder
        # self.deconv1 = UpSampling(2048, 1024)
        # self.bn1     = nn.BatchNorm2d(1024)
        
        # self.deconv2 = UpSampling(1024, 512)
        # self.bn2     = nn.BatchNorm2d(512)
        
        # self.deconv3 = UpSampling(512, 256)
        # self.bn3     = nn.BatchNorm2d(256)
        
        # self.deconv4 = UpSampling(256, 128)
        # self.bn4     = nn.BatchNorm2d(128)
        
        # self.deconv5 = UpSampling(128, 64)
        # self.bn5     = nn.BatchNorm2d(64)
        
        # ## residual (input to decoder)
        # self.residual = nn.Sequential(
        #     nn.Conv2d(3, 1, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(1)
        # )
                
        # self.seg_head = nn.Sequential(
        #     nn.Conv2d(64, 64, 3, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(True),
        #     nn.Dropout(0.5),
        #     nn.Conv2d(64, 32, 3, padding=1, bias=False),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(True),
        #     nn.Dropout(0.5),
        #     nn.Conv2d(32, self.C, 1),
        # )
            
    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,C,H,W)), C is the number of classes for segmentation.
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        
        
        ### 1. normalize
        x_normalized = (x - x.mean()) / torch.sqrt(x.var())
        
        ### 2. ResNest encoder -> list of outputs
        feature_maps = self.feature_encoder(x_normalized)
            
        ### 3. decoder with skipped connection    
        out = self.upsamling_decoder(feature_maps)
        
        ### 4. Residual connection
        out += self.residual(x_normalized)
        
        semantic_seg_scores = self.seg_head(out)
        
        return semantic_seg_scores
    
        # ### 3. decoder with skipped connection    
        # out = self.deconv1(feature_maps[3])
        # out = self.bn1(out + feature_maps[2])
        
        # out = self.deconv2(out)
        # out = self.bn2(out + feature_maps[1])
        
        # out = self.deconv3(out)
        # out = self.bn3(out + feature_maps[0])
        
        # out = self.deconv4(out)
        # out = self.bn4(out)
        # # out = self.bn4(out + feature_maps[0])
        
        # out = self.deconv5(out)
        # out = self.bn5(out)
        
        # ### 4. Residual connection
        # out += self.residual(x_normalized)
                
        # return self.seg_head(out)
    
class FCN_MT(torch.nn.Module):
    def __init__(self):
        """
        Your code here.
        Hint: The Multi-Task FCN needs to output both segmentation and depth maps at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        super().__init__()
        self.C = 19      

        self.feature_encoder = ResNetEncoder(skip_connect=True)
        
        ## decoder
        self.upsamling_decoder = UpSamplingDecoder()
        
        ## residual (input to decoder)
        self.residual = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1)
        )
                
        self.seg_head = FCN_PredictionHead(self.C)
        
        self.depth_head = FCN_PredictionHead(1)
        # self.depth_head = nn.Sequential(
        #     FCN_PredictionHead(1),
        #     nn.ReLU(True)
        # )
            

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,C,H,W)), C is the number of classes for segmentation
        @return: torch.Tensor((B,1,H,W)), 1 is one channel for depth estimation
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        ### 1. normalize
        x_normalized = (x - x.mean()) / torch.sqrt(x.var())
        
        ### 2. ResNest encoder -> list of outputs
        feature_maps = self.feature_encoder(x_normalized)
            
        ### 3. decoder with skipped connection    
        out = self.upsamling_decoder(feature_maps)
        
        ### 4. Residual connection
        out += self.residual(x_normalized)
        
        semantic_seg_scores = self.seg_head(out)
        depth_estimate_scores = self.depth_head(out)
        
        return semantic_seg_scores, depth_estimate_scores



class SoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SoftmaxCrossEntropyLoss, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Your code here
        Hint: inputs (prediction scores), targets (ground-truth labels)
        Hint: Implement a Softmax-CrossEntropy loss for classification
        Hint: return loss
        """
        
        exps = torch.exp(inputs)
        
        log_sofmax_prob = torch.log(exps) - torch.log(torch.sum(exps, dim=1, keepdim=True))
        # log_sofmax_prob = torch.log(
        #     exps / (torch.sum(exps, dim=1, keepdim=True) + 1e-15)
        # )

        targets_onehot = _one_hot(targets, exps.shape[1])
        
        cross_entropy_loss = -torch.sum(targets_onehot * log_sofmax_prob) / exps.shape[0]
        
        return cross_entropy_loss

# ---Helper Functions---

def _getBackboneLayers(pretrained_model):
    backbone = resnet50(weights=PRETRAINED_WEIGHTS)
    return [layer for layer in backbone.children()]

def _one_hot(x, n):
    # onehont_encoding = torch.eye(n).cuda(x.get_device())
    # onehot = onehont_encoding[x]
    return (x.view(-1, 1) == torch.arange(n, dtype=x.dtype, device=x.device)).int()
        
if __name__ == "__main__":
    x = torch.rand((2, 3, 64, 64))
    cnn = CNNClassifier()
    y = cnn.forward(x)