import os
import cv2


with open("./HAC/train/train.txt","w") as f:

    path_name = os.listdir("./HAC/train/input")
    for path in path_name:
        f.write('/train/input/'+path+' '+'/train/label/'+path+'\n')


with open("./HAC/test/test.txt","w") as f:

    path_name = os.listdir("./HAC/test/input")
    for path in path_name:
        f.write('/test/input/'+path+' '+'/test/label/'+path+'\n')

