#! /usr/bin/python3
import cv2
import os
import glob
root=os.getcwd()
folder='video'
vidiofold=os.path.join(root,folder)
videolist=glob.glob(vidiofold+'/*.mp4')
for i in videolist:
    vidcap = cv2.VideoCapture(i)
    name=i.split('/')[-1].split('-')[0].strip()
    os.makedirs(os.path.join(root,name))
    with open(os.path.join(os.path.join(root,name),'meta.txt'), 'w') as f:
        f.write(str(int(vidcap.get(cv2.CAP_PROP_FOURCC)))+'\n')
        f.write(str(vidcap.get(cv2.CAP_PROP_FPS))+'\n')
    success,image = vidcap.read()
    
    count = 0
    success = True
    while success:
        cv2.imwrite(os.path.join(os.path.join(root,name),"%05d.bmp" % count), image)     # save frame as PNG file
        success,image = vidcap.read()
        #print('Read a new frame: ', success)
        count += 1

