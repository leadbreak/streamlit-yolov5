import glob
import shutil
import os
import json

def main(l_path):
    cnt = 0
    dir = str(l_path)
    dir2 = l_path.replace('label','yolov5')
    names = [x.replace("\n", "") for x in open(f"{dir}/classes.names").readlines()]
    if not os.path.exists(dir2):
        os.mkdir(dir2)
    labelList = glob.glob(f"{dir}/*.json")
    for label in labelList :
        target = json.load(open(label))
        with open(f"{os.path.join(dir2, os.path.split(label)[1].replace('.json', '.txt'))}", 'w') as f :    
            for i in range(len(target['annotations'][0]['result'])) :
                text = " ".join([str(names.index(target['annotations'][0]['result'][i]['value']['rectanglelabels'][0])), # label
                                str((target['annotations'][0]['result'][i]['value']['x']+target['annotations'][0]['result'][i]['value']['width']/2)/100), # x
                                str((target['annotations'][0]['result'][i]['value']['y']+target['annotations'][0]['result'][i]['value']['height']/2)/100), # y
                                str(target['annotations'][0]['result'][i]['value']['width']/100), # width
                                str(target['annotations'][0]['result'][i]['value']['height']/100)]) # height
                f.write(text+"\n")
            f.close()

    classes = len(names)
    shutil.copy(glob.glob(f"{dir}/*.names")[0], f"{dir2}/classes.names")