# COMP4901V: Deep Perception, Localization, and Planning for Autonomous Vehicles

## Description
This course will deliver a knowledge base for students to understand fundamentals in autonomous vehicles under large-scale contexts within deep learning, covering three core modules: <br>
1. **Perception** \
Techniques for visual understanding of images and videos captured from RGB/LiDAR cameras.
   - object recognition
   - 2D/3D detection
   - semantic/instance segmentation
   - motion and flow estimation
   - multi-task perception

1. **Localization**
    - camera and geometry models
    - stereo and monocular depth estimation
    - 3D scene reconstruction
    - visual odometry, SLAM
    - multi-sensor fusion
  
2. **Prediction and planning**
   - pedestrian/vehicle trajectory prediction
   - motion/path planning
   - reinforcement learning in self-driving.

## Project Assignment 1
ILO:
- Dataset loader implementation
- Model design and implementation in Pytorch
- Model training pipeline implementation
- Model hyper-parameter tuning
- Model evaluation
- Result visualization
### Part a - CNN Image Classification
Implement a CNN classifier to train and evaluate on a VehicleClassification Dataset, using ResNet as Backbone.

### Part b - Multi-Task Dense Prediction
Implement a FCN model joinly trained on both semantic segmentation and depth estimation task.