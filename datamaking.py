import numpy as np
import cv2
import os
'''os.chdir('train/zhi')
path='10.png'
img=cv2.imread(path)
dst = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
h,w=np.shape(dst)
for i in range(h):
    for j in range(w):
        if dst[i][j]<100:
            dst[i][j]=0
        #else:dst[i][j]=0

cv2.imshow('grey'+path,dst)
cv2.waitKey(0)
cv2.imwrite('grey'+path,dst)
cv2.destroyAllWindows()'''
classname=['贝','之','其','出','在','大','小','我','是','月','石','雨','河','华','将','文']
source='train';target='x_train'
#source = np.unicode(source, 'utf-8')
for i in range(12):
    os.makedirs(os.path.join(target,str(i)))
for dir in os.listdir(source):
    dirname=os.path.join(source,dir)
    #dirname=np.unicode(dirname, 'utf-8')
    dstname=os.path.join(target,dir.split('_')[0])
    for j,file in enumerate(os.listdir(dirname)):
        impath=os.path.join(dirname,file)
        img=cv2.imread(impath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img=cv2.resize(img,(32,32))
        newfile=os.path.join(dstname,str(j)+'.jpg')
        cv2.imwrite(newfile,img)
