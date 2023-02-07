# YOLOv5 originally created by ultralytics on github using pytorch

# Guide on using yolov5

## CUDA / CuDNN

### Make sure your CUDA version is either 11.1 or 11.2, either should function properly when paired with the PyTorch version downloaded below.

### Tensorflow

### Tensorflow should be installed beforehand. Any version should work but with the Ubuntu 18.04 environment 2.3.0rc0 is preferred.

## Install rest of the required libraries

```
pip install -r requirements.txt
```

## Install pytorch version that is compatible with our CUDA build

```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu111
```

## Further guidance is provided within the next README inside of the yolov5 folder.