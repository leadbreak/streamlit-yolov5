# YOLOv5 pre-trained weights

걍 여기 참고하셈 ㄱㄱ:
https://github.com/ultralytics/yolov5#pretrained-checkpoints
https://github.com/ultralytics/yolov5/releases/tag/v5.0
쉽네

There are 10 different pre-trained weights available for transfer learning. 

5 different sizes of 2 different types. Sizes are : "n" for nano, "s" for small, "m" for medium, "l" for ~~take this L~~ large, and "x" for Xtra large (probably). And then there are 2 versions of these. There are the original P5 models and the P6 model. The P5 models consist of 3 output layers at strides 8, 16, 32 and are generally smaller and faster, while the P6 models have a 4th output layer at stride 64 and are slower and heavier but is better at detecting larger objects. P6 weights have a "6" after the model size. P6 weights are better trained at higher resolution so was trained at 1280 pixels, while the P5 models were trained on 640 pixels for speed.

The difference in the sizes of the models, other than the performance, of course, where the larger it is the better it performs, is the depth of its layers. The larger the weight, the deeper the layers of it. These weights should be used when training small to medium dataset, while for larger datasets, these weights may be unnecessary and increase training time.