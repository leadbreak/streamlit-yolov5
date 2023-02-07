from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

import uvicorn
from starlette.responses import JSONResponse
import subprocess as subp
import sys, requests
app = FastAPI()


class trainItem(BaseModel):
    projectId : int
    traindatasetId : int
    modelId : str
    dataPath : Optional[str] = None
    labelPath : Optional[str] = None
    modelPath : Optional[str] = None

class miniItem(BaseModel):
    projectId : int
    traindatasetId : int
    modelId : str


@app.get("/")
async def root() :
    return {"message":"Hello World"}


@app.post("/train/start")
async def train(item:trainItem) :

    """
    curl -X 'POST' \
  'http://127.0.0.1:8000/train/start' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "projectId": 1,
  "traindatasetId":1,
  "modelId": "yolo",
  "dataPath": "/home/wfs/projects/1/traindataset/4/data",
  "labelPath": "/home/wfs/projects/1/traindataset/4/label",
  "modelPath": "/home/wfs/projects/1/models/yolo/"
}'
    """ 

    subp.Popen(['streamlit', 'run', '/home/wfs/models/nets/yolov5/st_learn.py', '--', 
                f'--projectId={item.projectId}',
                f'--traindatasetId={item.traindatasetId}',
                f'--modelId={item.modelId}',
                f'--dataPath={item.dataPath}',
                f'--labelPath={item.labelPath}',
                f'--modelPath={item.modelPath}'])



@app.post("/train/stop")
async def stop(item:miniItem) :

    """
    curl -X 'POST' \
  'http://127.0.0.1:8000/train/stop' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "projectId": 1,
  "traindatasetId":1,
  "modelId": 1
}'
    """ 

    open(f"/home/wfs/projects/{item.projectId}/traindataset/{item.traindatasetId}/models/{item.modelId}/stop.txt", "w").close()



@app.post("/test/self")
async def test(item:miniItem) :

    """
    curl -X 'POST' \
  'http://127.0.0.1:8000/test/self' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "projectId": 1,
  "traindatasetId":1,
  "modelId": 1
}'
    """ 

    subp.Popen(['streamlit', 'run', '/home/wfs/models/nets/yolov5/st_test.py', '--', 
                f'--projectId={item.projectId}',
                f'--traindatasetId={item.traindatasetId}',
                f'--modelId={item.modelId}'])



@app.post("/test/free")
async def test(item:miniItem) :

    """
    curl -X 'POST' \
  'http://127.0.0.1:8000/test/free' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "projectId": 1,
  "traindatasetId":1,
  "modelId": 1
}'
    """ 

    subp.Popen(['streamlit', 'run', '/home/wfs/models/nets/yolov5/st_test.py', '--', 
                f'--projectId={item.projectId}',
                f'--traindatasetId={item.traindatasetId}',
                f'--modelId={item.modelId}',
                f'--testMode=freeTest'])
    


if __name__ == "__main__" :
    uvicorn.run("main:app", reload=True)