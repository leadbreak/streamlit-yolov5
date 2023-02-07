# Models files

These files are used by the main set of codes to build the model.

## common.py

Used for common modules and custom layers. Includes depth-wise convolution, depth-wise transpose convolution, transformer layer, bottleneck layer etc.

## experimental.py

Contains experimental modules such as sum layer that calculates the weighted sum of 2 or more layers and mixed depth-wise convolutions.

## yolo.py

Used for creating the yolo model using the modules from common and experimental.

## tf.py

Tensorflow, keras and tflite version of yolov5, used for building a yolov5 model using keras layers and modules instead of pytorch modules to allow for compatability with keras.