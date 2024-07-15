# Intel-Unnati-Programme
Project submission for Intel Unnati Training Programme 2024, Problem statement - Detect Pixelated image and correct it.

## Overview
This repository contains a model designed to detect and correct pixelated images. The model leverages deep learning techniques to identify pixelation artifacts and restore the image to a higher quality.

## Introduction
Pixelation often occurs when images are resized, compressed, or transmitted with a loss of quality. This project aims to address this issue by using a deep learning model that not only detects but also corrects pixelated images.

## Features
- **Detection:** Identify pixelated regions within an image.
- **Correction:** Restore pixelated regions to improve image quality.
- **End-to-End Pipeline:** An integrated pipeline for detection and correction.
- **Configurable:** Easy to adjust parameters and settings for different use cases.

## Architecture
The model consists of two main components:
1. **Pixelation Detection Model:** A convolutional neural network (CNN) that identifies pixelated regions.
2. **Pixelation Correction Model:** An image super-resolution model that corrects the identified pixelated regions.

## Pixelation Detection Model
- Uses a CNN architecture for detecting pixelated areas.
- Trained on a dataset with labeled pixelated and non-pixelated regions.

### Pixelation Correction Model
- Utilizes an advanced super-resolution technique.
- Enhances the quality of the detected pixelated regions.

 
## Dataset Folder Structure
Organize your dataset with the following structure:

- **dataset**
  - **train**
    - Pixelated
    - Non_Pixelated
  - **test**
    - Pixelated
    - Non_Pixelated

