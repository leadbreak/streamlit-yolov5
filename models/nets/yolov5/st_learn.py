"""
To use argparse in streamlit run,
should write '--' one more.
ex)  streamlit run st_learn.py -- --projectId=1 --traindatasetId=4 --modelId=1 --dataPath=/home/wfs/projects/1/traindataset/4/images --labelPath=/home/wfs/projects/1/traindataset/4/label --modelPath=/home/wfs/projects/1/traindataset/4/models/yolo/
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, time
from glob import glob
import subprocess as subp
import sys
from PIL import Image
import re
import altair as alt

import signal
import psutil, shutil, json, math
import requests

import argparse

import functions.data as dta
import functions.ttxt as ttxt
import functions.convert_lab as clab

parser = argparse.ArgumentParser(description='start train code')

parser.add_argument('--projectId', default=1)
parser.add_argument('--traindatasetId', default=4)
parser.add_argument('--modelId', default='yolo')
parser.add_argument('--dataPath', default='/home/wfs/projects/1/traindataset/4/data')
parser.add_argument('--labelPath', default='/home/wfs/projects/1/traindataset/4/label')
parser.add_argument('--modelPath', default='/home/wfs/projects/1/traindataset/4/models/yolo/' )

args = parser.parse_args()

st.session_state.projectId = args.projectId
st.session_state.traindatasetId = args.traindatasetId
st.session_state.modelId = args.modelId
st.session_state.dataPath = args.dataPath # '/home/wfs/projects/1/traindataset/4/data'
st.session_state.labelPath = args.labelPath # '/home/wfs/projects/1/traindataset/4/label' 
st.session_state.modelPath = args.modelPath # '/home/wfs/projects/1/traindataset/4/models/yolo/' 
if st.session_state.modelPath[-1] == os.sep:
    st.session_state.modelPath = st.session_state.modelPath[:-1]
if st.session_state.dataPath[-1] == os.sep:
    st.session_state.dataPath = st.session_state.dataPath[:-1]
if st.session_state.labelPath[-1] == os.sep:
    st.session_state.labelPath = st.session_state.labelPath[:-1]
if "nets/yolov5" not in os.getcwd() :    
    os.chdir("/home/wfs/models/nets/yolov5/")

if __name__ == "__main__" :
    st.set_page_config(
            page_title="in2wiser",
            page_icon="🧊",
            layout="wide",
            menu_items={
                'Get Help': 'https://github.com/leadbreak',
                'Report a bug': "https://github.com/leadbreak",
                'About': "### This is a test app for the new skill!"
            }
        )

    st.markdown(
        """
            <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                footer:after {
                    content : 'Made by in2wise';
                    visibility:visible;
                    display: block;
                    position: relative;
                    #background-color: red;
                    padding: 5px;
                    top: 2px;
                }
                .reportview-container .main .block-container{{                   
                    padding-top: 0rem;
                }}
            </style>
            """,
            unsafe_allow_html=True,
            )
            
    # Project ID와 Model ID로 url 열기
    st.experimental_set_query_params(
        process=['train'],
        project=st.session_state.projectId,
        traindataset=st.session_state.traindatasetId,
        model=st.session_state.modelId
        )


    #   #   #   #   #   #   #   #   #   #   #   #   # 
    # PROCESS1 : data & label 데이터 가져오고, cfg.yaml 등 생성하는 사전 작업
    # PROCESS2 : 모델 학습하며 로그 떨어뜨리기
    # PROCESS3 : 모델 경량화
    # PROCESS4 : 모델 테스트
    #   #   #   #   #   #   #   #   #   #   #   #   #
    with st.spinner('Processing Data'):

        # PROCESS1 : 사전 작업 진행
        process1 = subp.Popen([f'{sys.executable}',f"./check_del.py",str(st.session_state.modelPath)], stdout=subp.PIPE)
        while process1.poll() is None:
            pass
        process1.kill()
        if not os.path.exists(st.session_state.modelPath):
            os.makedirs(st.session_state.modelPath)

        ## STEP 01. 학습에 필요한 데이터를 모으고, 작업 결과를 처리할 폴더 경로는 생성되어 있다고 가정?
        ttxt.main(st.session_state.dataPath, st.session_state.labelPath, st.session_state.modelPath)

        ## STEP 02. Train Info(epoch, batch)를 반영 or 지정
        n_epochs = 1000

        ## STEP 03. classes.names 등을 활용해 cfg.yaml 생성        
        clab.main(st.session_state.labelPath)
        st.session_state.labelPath = st.session_state.labelPath.replace('label', 'yolov5') # begin using labels folder instead of the old one
        dta.main(st.session_state.labelPath, st.session_state.modelPath)
        
        ## STEP 04. SubProcess를 이용해 Train을 진행
        loss = [np.nan for _ in range(n_epochs+1)]
        maP = [np.nan for _ in range(n_epochs+1)]
        log = []
        epoch_dur = []

        process3_empt = st.empty()
        graf = st.container()
        col = st.empty()
        loges = st.empty()
        loges_title = st.empty()
        act_graf = st.empty()
        if 'check_d' not in st.session_state:
            st.session_state.check_d = 1
        try :
            os.remove(f'{st.session_state.modelPath}stop.txt')
        except :
            pass
        epoch = 0
        max_dur = 0
        st.session_state.stop_code = os.path.exists(f'{st.session_state.modelPath}stop.txt')
        if os.path.exists(f'{st.session_state.modelPath}/log.csv') :
            os.remove(f'{st.session_state.modelPath}/log.csv')
        
        time.sleep(1)
    
    # PROCESS2 : 모델 학습
    process2 = subp.Popen([f'{sys.executable}',f"./train.py", f"--weights=yolo_weights/yolov5s.pt", f"--img=416", f"--batch=-1", f"--epochs={n_epochs}", f"--project={st.session_state.modelPath}",f"--data={st.session_state.labelPath}/cfg.yaml",f"--device=0",f"--patience=300", f"--exist-ok"], stdout=subp.PIPE)
    while process2.poll() is None:
        if st.session_state.check_d == 1:
            while True:
                with st.spinner('Loading Data...'):
                    linen = process2.stdout.readline()
                    linen = linen.decode('utf-8')
                    if 'mAP' in linen:
                        st.session_state.check_d = 0
                        break
        st.session_state.stop_code = os.path.exists(f'{st.session_state.modelPath}stop.txt')
        line = process2.stdout.readline()
        line = line.decode('utf-8')
        x = np.arange(n_epochs)
        
        if 'mAP' in line:   
            lines = re.split(':|,| ', line)
            loss[int(lines[3])] = float(lines[8])
            maP[int(lines[3])] = float(lines[13])
            df = pd.DataFrame({'Epochs':x,'loss':loss[1:], 'maP':maP[1:]})
            cur_log = ' |&nbsp;&nbsp; Epoch: '+ str(int(lines[3])+1) + '&nbsp;&nbsp;|&nbsp;&nbsp; Loss: ' + lines[8][0:6] + '&nbsp;&nbsp;|&nbsp;&nbsp; maP: ' + str(maP[int(lines[3])-1]) + "&nbsp;&nbsp;|&nbsp;&nbsp;-&nbsp;&nbsp;" + lines[19][0:6] + ' sec' 
            
            epoch_dur.append(float(lines[19]))
            log.append(cur_log)

            if int(lines[3])+1 == n_epochs:
                spent_t = int(sum(epoch_dur))                    
                loges.success('Training Completed&nbsp;&nbsp;|&nbsp;&nbsp;Elapsed Time: ' + str(spent_t))

            else:
                spent_t = int(sum(epoch_dur))                    
                max_dur = max(max([x for x in epoch_dur[-1:-20:-1]]), max_dur)
                rem_dur = int(max_dur*(n_epochs - int(lines[3])))
                loges.write('Elapsed Time: ' + str(spent_t) + ' seconds &nbsp;&nbsp;&nbsp;&nbsp; Estimated Remaining time: ' + str(rem_dur) + ' seconds')

            s = '\n\n\n'.join([x for x in log[-1:-100:-1]])

            with act_graf.container(): 
                st.markdown(s)
            base = alt.Chart(df).encode(
                alt.X('Epochs', axis = alt.Axis(title = 'Epoch'))
            )
            c = base.mark_line().encode(
                alt.Y('loss', axis = alt.Axis(title = 'Loss'))
            )
            d = base.mark_line(color = 'red').encode(
                alt.Y('maP', axis = alt.Axis(title = 'maP (%)'))
            )
            f = alt.layer(c, d).resolve_scale(
                y = 'independent'
            )
            col.altair_chart(f, use_container_width=True)
            loges_title.write('Latest 100 Epochs summary:')

        st.session_state.stop_code = os.path.exists(f'{st.session_state.modelPath}stop.txt')
        if st.session_state.stop_code :
            with loges_title :
                st.error("Found Stop Request")
            os.remove(f'{st.session_state.modelPath}stop.txt')

            os.system('clear')
            parent = psutil.Process(process2.pid) # 부모 프로세스를 저장된 pid 로 로드한다
            childrens = parent.children(recursive=True) # 로드된 부모로 아이를 착출해 모든 아이 프로세스들을 저장한다
            for child in childrens: 
                child.send_signal(signal.SIGTERM) # 모든 아이 프로세스 종료
            time.sleep(1) 
            os.kill(process2.pid, signal.SIGTERM) # 부모 프로세스 종료
            time.sleep(1)
            
            os._exit(1) # 현재 프로세스 종료
    process2.kill()

    # PROCESS3 : 경량화
    process3 = subp.Popen([f'{sys.executable}',f"./export.py",  f"--data={st.session_state.labelPath}/cfg.yaml", f"--weights={st.session_state.modelPath}/weights/best.pt", f"--include=tflite", f"--img=416"], stdout=subp.PIPE)
    with loges_title :
        with st.spinner("Model Optimizing is on process...") :    
            while process3.poll() is None:
                line = process3.stdout.readline().decode('utf-8')
                if "Converting Tflite is Complete!" in line :
                    print("="*80)
                    print("\nModel Training & Optimizing are complete!\n")
                    print("="*80)
                    time.sleep(1)
                    process3.kill()
    process3.kill()

    # PROCESS4 : 테스트 코드
    process4 = subp.Popen([f'{sys.executable}',f"./detect.py", f"--weights={st.session_state.modelPath}/weights/best-fp16.tflite", f"--img=416", f"--conf=0.5", f"--data={st.session_state.labelPath}/cfg.yaml", f"--iou=0.25", f"--project={st.session_state.modelPath}", f"--name=results", f"--source={st.session_state.dataPath}"], stdout=subp.PIPE)
    while process4.poll() is None:
        st.session_state.stop_code = os.path.exists(f'{st.session_state.modelPath}stop.txt')
        if st.session_state.stop_code :
            with loges_title :
                st.error("Found Stop Request")
            os.remove(f'{st.session_state.modelPath}stop.txt')

            os.system('clear')
            parent = psutil.Process(process4.pid) # 부모 프로세스를 저장된 pid 로 로드한다
            childrens = parent.children(recursive=True) # 로드된 부모로 아이를 착출해 모든 아이 프로세스들을 저장한다
            for child in childrens: 
                child.send_signal(signal.SIGTERM) # 모든 아이 프로세스 종료
            time.sleep(1) 
            os.kill(process4.pid, signal.SIGTERM) # 부모 프로세스 종료
            time.sleep(1)
        pass
    process4.kill()

    act_graf.success("Model Training & Optimizing are complete!")
    
    successEmpty = st.empty()
    for i in reversed(range(5))  :
        with successEmpty :
            st.success(f"{i+1}초 뒤 프로세스가 종료됩니다.")
        time.sleep(1)

    with successEmpty :
        st.success(f"프로세스가 종료되었습니다.")


    os._exit(1)        
