# unet-box-detection-bin-picking-robot

## Description:

This repository implements the uNET architecture for detecting boxes, designed specifically for a robot-based bin-picking system. The project focuses on the computer vision aspect, particularly on identifying boxes within a given scene. The robust implementation of the uNET architecture facilitates precise box detection, enabling seamless integration with robotic systems for efficient bin picking tasks.

## Key Features:

Utilizes uNET architecture for accurate box detection. Specifically tailored for robot-based bin-picking systems. Designed to enhance automation and efficiency in industrial settings. Usage:

Ideal for developers and engineers working on robotic systems. Provides a foundation for implementing box detection in industrial environments.

Contributions: Contributions are welcome to enhance the efficiency and versatility of box detection algorithms, enabling broader applications in industrial automation.

## Pre-trained Models:

# Semantic Segmentation:
You can download a pre-trained model for semantic segmentation from [here](https://drive.google.com/file/d/1--JQXFLKpjepVSc5hTFvlfGhqozkduLl/view). Place the model into the models/ directory.

## Inference On Google Colab:

For convenience, you can run the inference on Google Colab using our pre-trained models. Access the Colab notebook [here](https://colab.research.google.com/drive/1u_lYgVbUwJPmJK6ds2oMLnGOx2tA1W5n?usp=sharing).
You can see some example inference images from [here](https://github.com/KavinduJ2001/UNet-Implementation/tree/main/Inference%20Images).

# Instant Segmentation:
We also trained the model using instant segmentation, but we couldn't get satisfactory outputs from that. You can access the pre-trained model [here](https://drive.google.com/file/d/1-0aUVPD53Hw8CtVmLkNkqKZBWkLTfdwk/view?usp=share_link). The new dataset used for instant segmentation is available [here](). 

## Note: 

On macOS, .DS_Store files are automatically created by Finder to store folder view settings. In this project, it’s important to handle these files properly.
