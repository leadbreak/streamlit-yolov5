import os
import shutil
from glob import glob

def main(d_path, l_path, m_path):
    if d_path[-1] == os.sep:
        d_path = d_path[:-1]
    if l_path[-1] == os.sep:
        l_path = l_path[:-1]

    imgList = [os.path.join(d_path, os.path.split(x)[1].replace(".json", ".jpg")) for x in glob(f'{l_path}/*.json')]


    ## train.txt 생성
    with open(os.path.join(m_path, 'train.txt'), 'w') as f :
        tmp = [f.write(target+"\n") for target in imgList]
        f.close()
    del tmp