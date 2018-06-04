#! /usr/bin/python3
import cv2
import os,glob

root=os.getcwd()
join=os.path.join
dataset=join(root,'dataset')
folder=join(root,'vresult')
image_list=[img for img in os.listdir(folder)]
for i in image_list:
    temp=join(dataset,i)
    temp=join(temp,'meta.txt')
    with open(temp,'r') as f:
        foc=f.readline()
        fps=f.readline()
    #print(fps.strip(),foc.strip())
    im_list=glob.glob(join(folder,i+"/*bmp"))
    #print(type(im_list))
    im_list.sort()
    #print(im_list)
    frame = cv2.imread(im_list[0])
    height, width, layers = frame.shape
    fourcc =foc.strip()
    video = cv2.VideoWriter(join(folder,i)+'.mp4',int(fourcc),float(fps.strip()), (width,height))
    for image in im_list:
        video.write(cv2.imread(image))
    cv2.destroyAllWindows()
    video.release()
