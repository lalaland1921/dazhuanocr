from vertical_horizontal_simplecut import Cut
from inference import infer,cv2ImgAddText
import cv2

'''pic_path=input("请输入图片路径")
minThred=input("请输入阈值")'''
pic_path='./testimages/6.png'
minThred=0.3
img=cv2.imread(pic_path)
part_imgs=Cut(pic_path)
translate=infer(part_imgs,minThred)
img=cv2ImgAddText(img,translate,10,10)
cv2.imshow("the translated image:",img)
cv2.waitKey(0)