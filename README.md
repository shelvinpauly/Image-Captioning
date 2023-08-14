# Image Captioning with Vision Transformers

## Overview

This project implements an image captioning model using a vision transformer (ViT) architecture. The model takes an image as input and generates a descriptive caption for the image.

The model consists of three main components:

- A CNN backbone (ResNet34) to extract image features
- A transformer encoder to refine the image features
- A transformer decoder to generate the caption

The encoder-decoder transformer architecture allows the model to learn global relationships between parts of the image and generate detailed, relevant captions.

## Dataset

The model is trained and evaluated on the [MS COCO dataset](https://cocodataset.org/#overview). This contains over 120,000 images with 5 captions per image.

## Training

The model is implemented in PyTorch and trained for 10 epochs using cross-entropy loss and the Adam optimizer. Data augmentation techniques like random horizontal flipping, cropping, and color jittering are used during training.

The best model achieves a lower training loss compared to a baseline CNN + RNN model.

## Evaluation

The model generates more descriptive and accurate captions compared to the baseline model when evaluated on the COCO test set. It is also able to generalize reasonably well to out-of-domain images from the Flickr30k dataset.

## Usage

The trained model can be used to generate captions for new images:

```python
import model

img = load_image('example.jpg')
caption = model.generate_caption(img) 
print(caption)
```

## References

Relevant papers:

- [Image Transformer](https://arxiv.org/abs/1802.05751)
- [Meshed-Memory Transformer](https://arxiv.org/abs/1912.08226) 
- [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

[Project Report](report.pdf)
