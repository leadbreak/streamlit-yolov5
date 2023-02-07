# Utils

Contains various modules and libraries that are used for transforming, training and logging the training process.

## activations.py

As the name states, this contains the different activation functions for model to use. It contains the following activation functions:

- Sigmoid Linear Unit (SiLU)
- Hard-SiLU
- Mish
- Memory efficient Mish
- Flexible ReLU
- ACON
- MetaACON

These are unused, but if one wanted to use them, they could replace the existing activation function in the common.py layer modules.

## augmentations.py

Used for image augmentations. Includes:

- HSV color-space augmentation
- Histogram equalization for bgr images
- Resize and padding for stride compatibility
- Random perspective
- Cut-out etc.

## autoanchor.py

Anchor related modules for the image dataset. Computes the anchors using kmeans, and if anchors are given it checks the anchors to see if they are valid, and if not recomputes or adjusts them.

## autobatch.py

Automatically creates a batch size to minimize memory usage.

## benchmarks.py

Runs the YOLOv5 benchmarks on any of the model formats supported. ie. pt, tflite, keras.saved_model etc.

## callbacks.py

Contains the callback utils, such as registering a new action upon callback, returning all the registered actions or looping through the registered actions and firing all callbacks.

## downloads.py

Used to download any items that are not available locally, such as the coco dataset or any weights that needs to be used for transfer learning. Mostly unnecessary.

## general.py

General utils, such as checking requirements python and library versions, image size, .yaml file, dataset integrity, the environment, logger and a lot more.

## loss.py

Computes the loss for training. Has BCE with logits and Focal loss. Calculates the class, box and object loss.

## metrics.py

Computes various metrics for judging the model performance, such as fitness, precision, recall, confusion matrix and average precision per class.

## plots.py

Used for creating the plots within the training informations.