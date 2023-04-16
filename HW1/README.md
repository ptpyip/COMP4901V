# Project Assignment 1
ILO:
- Dataset loader implementation
- Model design and implementation in Pytorch
- Model training pipeline implementation
- Model hyper-parameter tuning
- Model evaluation
- Result visualization
## Part a - CNN Image Classification
Implement a CNN classifier to train and evaluate on a VehicleClassification Dataset, using ResNet as Backbone.

## Part b - Multi-Task Dense Prediction
Implement a FCN model joinly trained on both semantic segmentation and depth estimation task.

## Results

### Visualization of Multi-Task FCN Results
(Top-to-bottom: img -> depth estimate -> GT detph -> semantic segmentation -> GT semantic label)
![CleanShot 2023-04-16 at 19 08 08](https://user-images.githubusercontent.com/18398848/232305254-3e3a9fe3-cce1-4c65-9932-4a319b80c388.png)


### Traing of CNN Image Classification
![image](https://user-images.githubusercontent.com/18398848/232303797-bb40d03c-c22b-4bde-a49e-3900c3646d52.png)

![image](https://user-images.githubusercontent.com/18398848/232303831-0c520b10-2dc2-4390-85d0-752d7ed7b488.png)

### Result of CNN Image Classification
![image](https://user-images.githubusercontent.com/18398848/232304216-a037a59a-7bf0-41ed-97dc-589d74eca6ab.png)

### Traing of Single Task FCN Segmentation
![image](https://user-images.githubusercontent.com/18398848/232304428-c87f21ee-5da5-48f0-8cc0-2f9c8d1e59cf.png)

![image](https://user-images.githubusercontent.com/18398848/232304545-c064039a-68b5-4f05-9d43-0dcc57741799.png)

### Result of Single Task FCN Segmentation
![image](https://user-images.githubusercontent.com/18398848/232304673-5123787f-8e0f-4077-bfb9-174f5d72e82e.png)

### Traing of Multi-Task FCN
![image](https://user-images.githubusercontent.com/18398848/232304866-9e4e4307-3c1b-436a-b353-9e90cf3157c0.png)

![image](https://user-images.githubusercontent.com/18398848/232304902-e2b0e3d9-e4e6-4157-9970-574e38c5c62b.png)


### Result of Multi-Task FCN
![image](https://user-images.githubusercontent.com/18398848/232304940-b2799f13-d0f0-4cdf-acec-fd747040d8b8.png)
