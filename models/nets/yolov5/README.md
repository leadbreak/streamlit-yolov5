# Code Details

## Data formatting

The data should be formatted in a way where all the images are in its separate folder, named whatever, and annotations should be included within a folder called "labels", although this can be altered as long as its consistent between projects. For example, if you wanted the labels to be contained within "annotations" all projects then on should have its annotations included in the folder named "annotations". classes.names should be in the "labels" folder along with annotations and include only the classes in its own line. Then, train.txt is required outside of the images/labels folder, and should include the absolute paths to the images, ie. "/home/wfs/projects/1/trainsetdata/1/images/image.jpg" without the quotation marks.

## ttxt.py

When the data is given, it will not have train.txt, so run this code with the data and labels path and the models directory as inputs and it will create train.txt in the models directory with absolute paths in the directory along with the data and labels folder. 

Example:
``` 
python3 ttxt.py /home/wfs/projects/1/traindataset/4/images /home/wfs/projects/1/traindataset/4/label /home/wfs/projects/1/traindataset/4/models/1
```
input: Images and label directory
output: New data directory and train.txt

## convert_lab.py

The labels folder at first will be in .json file in the voc data format, so run this file passing in the label directory and it will create a new directory named labels that will have the data in yolo format as .txt.

Example:
``` 
python3 convert_lab.py /home/wfs/projects/1/traindataset/4/label
```
input: Label directory with .json files in voc format

output: Labels directory with .txt files in YOLO format

## data.py

With the data formatted as stated above, you can run "data.py", with the labels directory along with the models directory. This will create cfg.yaml which will include the path to the images and label folder, path to the images which will be linked using the train.txt created earlier, number of classes, and the classes themselves as a list. At the moment the validation and test images all use train.txt as we don't have enough data to split data, however, if needed or wanted it should be simple to create a separate val.txt and text.txt and include it instead of train.txt for their respective image paths.

Example:
``` 
python3 data.py /home/wfs/projects/1/traindataset/4/labels /home/wfs/projects/1/traindataset/4/models/1
```
input: Labels directory

output: cfg.yaml

## check_del.py

check_del.py takes in the model directory, and checks if it exists. If it does exist, it deletes it. Otherwise, nothing happens.

Example: 
``` 
python3 check_del.py /home/wfs/projects/1/traindataset/4/models/1
```
input: Model directory name

output: Nothing if instance doesn't exist, deleted instance if it exists

## train.py

train.py can take many arguments, but in st_learn.py I have added static arguments for necessary ones that won't change, and allowed for customizable inputs for some arguments. All relevant arguments for this code includes, a pre-trained weight, a config file, path to the .yaml file, epoch size, batch size, image transformation size, changing weights for different images, device selection, optimizer, number of workers, project name, instance name, patience, save-period and more that are less relevant. The pretrained weights are defaulted to the "s" yolo weight, which is the lightest of all the weights, config is unncessary, .yaml file will be directed to the path to the project name that was entered, epoch size is customizable, batch size is set to -1 which will allow for the code to automatically set batch size, image size is set to 416 per request of my superior, image weights are un-used, device is set to gpu 0, optimizer defaulted to SGD, 8 workers, project name and instance name should be inputted when running the shell script, or one can specify it but it should always be specified, patience is defaulted to 100 but I have set it to 300, save period is un-used and the other arguments are not too useful. When this code is ran, it will load the data according to the project name and the .yaml file under it, and throughout its epochs it will output object loss, class loss, mAP, precision, recall and epoch duration in the terminal. Throughout training, until the end, it will store the confusion matrix, F1 curve, tensorboard event, some batches for training and validation, along with the results csv and graphs, including box loss, object loss, class loss, precision, recall mAP@0.5 mAP@0.5:0.95 under the instance folder. Finally, the best and last weights will be saved under the weights folder along with the training information in the pt format.

Additional Notes: To load pretrained weights, you should add the extra argument "--weights=weights/yolov5*.pt". You can add any pre-trained weights in the pt format, but all of the pre-trained yolov5 weights are inside the weights folder.

Example:
``` 
python3 train.py --img 416 --batch -1 --epochs 9999999 --project /home/wfs/projects/3 --name test --data /home/wfs/projects/3/traindataset/1/3.yaml --device 0 --patience 300
```
input: image size, batch size, epoch size, project name, output/instance name, path to .yaml file, gpu device, and patience, pre-trained yolo weights.

output: training information and best and last weights in .pt format

## export.py

export.py is used to convert yolo weight to tflite. It is able to save it into many different formats, such as torchscript, onnx, tensorflow etc. but we will use tflite primarily as we intend to use the model on an edge device or whatever. Its arguments include, path to .yaml file, weight to convert, image size, and output model format. Path to .yaml is the same as train.py, weights is set to the best.pt weight using the project and instance name input into the shell script, image size should be equivalent to image size used to train the model so it's set to 416, and output format is obviously set to tflite. Other arguments are irrelevant for what we attempt to do here and exists for additional training or whatever. When this code is ran, it first loads the torch model, evaluates the model, create dummy inputs, convert it to tensorflow saved_model format, converts it to tflite and saves it to same weights folder as best.pt.

Example:

```
python3 export.py --data /home/wfs/projects/3/traindataset/1/3.yaml --weights /home/wfs/projects/3/test/weights/best.pt --include tflite --img 416
```
input: path to best.pt weight, .yaml file, output model format, and image size

output: best-fp16.tflite model

## detect.py

detect.py is obviously used as an inference code using the weights derived from the previous processes. Its arguments include, the weight/model to be used, path to the images to be inferred, image size, confidence threshold, iou threshold, path to the yaml file, and the project and instance name to save the results to. There are other arguments such as line thickness, hiding labels, saving results and confidences as txt file, but are omitted due to lacking relevance. The model is set to the tflite model within the project and instance name entered, image size should be the same as input image sizes so it's set to 416, path to the images are set to the training images directory, confidence threshold is set to 0.5, iou threshold is 0.25, path to .yaml file is same as the previous 2 codes, project directory is set to the instance name as we want to save it under the instance and not directly under the project, and directory name is set to "results". This is lumped into the shell script just to simplify testing out the model on the trianing images, but if an individual wished to test out the model on a separate set of images, some arguments would need to be altered. The only arguments that needs to be changed when running this code by itself it obviously the images source folder, iou and confidence threshold if need be, and the instance name unless you wish to save it under results.

Example:

``` 
python3 detect.py --weights /home/wfs/projects/3/test/weights/best-fp16.tflite --img 416 --conf 0.5 --data /home/wfs/projects/3/traindataset/1/3.yaml --iou 0.25 --project /home/wfs/projects/3/test --name results --source /home/wfs/projects/3/traindataset/1/images
```
input: Model, image size, confidence threshold, iou threshold, .yaml file, project and output directory name, and directory to images or an image.

output: Directory of images with prediction boxes/classes and confidence scores

## st_learn.py

Ties up all of the codes mentioned above together. It takes in all of the ID's along with the data, label and the model path as input. It first checks if the model directory exists, then if it does deletes it, then creates the model path again. Then it proceeds to creating the train.txt and cfg.yaml, after which it converts the .json annotations to .txt labels under the folder "yolov5". It then trains on the data, converts the model to tflite and creates detections on training set. Boom ez.

```
streamlit run st_learn.py -- --projectId=1 --traindatasetId=3 --modelId=1 --dataPath=/home/wfs/projects/1/traindataset/3/images --labelPath=/home/wfs/projects/1/traindataset/3/label --modelPath=/home/wfs/projects/1/traindataset/3/models/1 
```

## st_test.py

Allows for testing of the model using either a directory of images or uploading custom images. There are two modes, self test and free test. in self test, you enter a directory of images that you want to test out the model on and it will automatically make inference on those images and display it with a fixed confidence and iou threshold. With free test, you don't need to enter a directory path, and just input a model path. When you input a model path, you will be greeted with 2 slide bars that will allow you to control the confidence and iou threshold. Then in the middle of the screen you can upload an image or images from your computer that the model will make inference on right away. Then you can use the slide bars to control the confidence and iou threshold to see how the bounding boxes change when you do.

```
streamlit run st_test.py -- --testMode=freeTest --modelId=/home/wfs/projects/1/traindataset/3/models/1

streamlit run st_test.py -- --testMode=selfTest --modelId=/home/wfs/projects/1/traindataset/3/models/1 --traindatasetId=/home/wfs/projects/1/traindataset/3/images
```

## Additional Code

## val.py

Mainly used by train.py to calculate mAP at the end of an epoch, but is also capable being ran on its own to make inference and make other evaluations and validations.

## predtflite.py

Used for finding the coordinates in a dictionary format that the data was originally given in. It saves the data to the test directory within the current working directory in .txt files. It contains the top left corner coordinates with width and height, and is divided by image width and heigh respectively and multiplied by 100. For example x_new = x_old/1920 * 100, y_new = y_old/1280 * 100.