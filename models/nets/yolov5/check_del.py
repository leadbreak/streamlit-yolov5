import os
import sys
import shutil

dir = sys.argv[1]
if os.path.exists(dir):
    shutil.rmtree(dir)
    print("Duplicate removed")