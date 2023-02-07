import os
import yaml
import sys

def main(l_path, m_path):
    dir = l_path
    dir2 = l_path.replace('label', 'yolov5')
    f = open(os.path.join(dir, 'classes.names'), 'rb')
    name = f.readlines()
    for x in range(len(name)):
        name[x] = name[x].decode().strip()

    data = {
        'path':m_path,
        'train':'train.txt',
        'val':'train.txt',
        'test':'train.txt',
        'nc':len(name),
        'names':name
    }

    with open(os.path.join(dir2,'cfg.yaml'), 'w') as q:
        yaml.safe_dump(data, q, default_flow_style=None)