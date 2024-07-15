
# Intel-Unnati-Programme
Project submission for Intel Unnati Training Programme 2024, Problem statement - Detect Pixeleted image and correct it.
# Problem Statement: Detect Pixeleted Image and Correct It

## Overview
This repository contains a model designed to detect and correct pixeleted images. The model leverages deep learning techniques to identify pixelation artifacts and restore the image to a higher quality.

## Introduction
Pixelation often occurs when images are resized, compressed, or transmitted with a loss of quality. This project aims to address this issue by using a deep learning model that not only detects but also corrects pixeleted images.

## Features
- **Detection:** Identify pixeleted regions within an image.
- **Correction:** Restore pixeleted regions to improve image quality.
- **End-to-End Pipeline:** An integrated pipeline for detection and correction.
- **Configurable:** Easy to adjust parameters and settings for different use cases.

## Architecture
The model consists of two main components:
1. **Pixelation Detection Model:** A convolutional neural network (CNN) that identifies pixeleted regions.
2. **Pixelation Correction Model:** An image super-resolution model that corrects the identified pixeleted regions.

## Pixelation Detection Model
- Uses a CNN architecture for detecting pixeleted areas.
- Trained on a dataset with labeled pixeleted and non-pixeleted regions.

## Pixelation Correction Model
- Utilizes upscaling technique.
- Enhances the quality of the detected pixeleted regions.

 
## Dataset Folder Structure (for detection model)
Organize your dataset with the following structure:

- **dataset**
  - **train**
    - Pixeleted
      - 1.jpg
    - Non_Pixeleted
      - 1.jpg
  - **test**
    - Pixeleted
      - 1.jpg
    - Non_Pixeleted
      - 1.jpg

## Dataset Folder Structure (for correction model)
Organize your dataset with the following structure:

- **datat**
  - **train**
    - X
      - 1.jpg
    - y
      - 1.jpg
  - **test**
    - X
      - 1.jpg
    - y
      - 1.jpg

# Inference
There are three scripts for inference.
1. inference.py
2. repeated-inference.py
3. video-inference.py

- run the `inference.py` script to correct a pixeleted image.
- run the `repeated-inference.py` script to correct a pixelated image by repeatedly feeding the pixeleted blocks to the correction model
- run the `video-inference.py` script to display your camera feed and use detection and correction models on each frame.

**Note**: The `inference.py` and `repeaded-inference.py` scripts saves the output image as `output.jpg` in the same folder.
