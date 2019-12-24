from PIL import Image
import numpy as np
import cv2

min_thresh = 2  # 字符上最少的像素点
min_range = 5  # 字符最小的宽度


def vertical(img_arr):
    h, w = img_arr.shape
    ver_list = []
    for x in range(w):
        ver_list.append(h - np.count_nonzero(img_arr[:, x]))
    return ver_list


def horizon(img_arr):
    h, w = img_arr.shape
    hor_list = []
    for x in range(h):
        hor_list.append(w - np.count_nonzero(img_arr[x, :]))
    return hor_list


def OTSU_enhance(img_gray, th_begin=0, th_end=256, th_step=1):#找到合适的阈值
    max_g = 0
    suitable_th = 0
    for threshold in range(th_begin, th_end, th_step):
        bin_img = img_gray > threshold#前景图片部分，即字
        bin_img_inv = img_gray <= threshold#背景图片部分
        fore_pix = np.sum(bin_img)#前景图片像素数量
        back_pix = np.sum(bin_img_inv)
        if 0 == fore_pix:#无字，跳出
            break
        if 0 == back_pix:
            continue

        w0 = float(fore_pix) / img_gray.size#前景图片像素数量所占比例
        u0 = float(np.sum(img_gray * bin_img)) / fore_pix#前景图片平均像素
        w1 = float(back_pix) / img_gray.size#背景图片像素所占比例
        u1 = float(np.sum(img_gray * bin_img_inv)) / back_pix#背景图片平均像素值
        # intra-class variance
        g = w0 * w1 * (u0 - u1) * (u0 - u1)#最大化这个
        if g > max_g:
            max_g = g
            suitable_th = threshold
    return suitable_th


def cut_line(horz, pic):
    begin, end = 0, 0
    w, h = pic.size
    cuts = []
    imglist=[]
    for i, count in enumerate(horz):
        if count >= min_thresh and begin == 0:
            begin = i
        elif count >= min_thresh and begin != 0:
            continue
        elif count <= min_thresh and begin != 0:
            end = i
            # print (begin, end), count
            if end - begin >= 2:
                cuts.append((end - begin, begin, end))
                begin = 0
                end = 0
                continue
        elif count <= min_thresh or begin == 0:
            continue
    #cuts = sorted(cuts, reverse=True)
    if len(cuts) == 0:
        return 0, 0,False
    '''else:
        if len(cuts) > 1 and cuts[1][0] in range(int(cuts[0][0] * 0.8), cuts[0][0]):
            return 0, False'''
    re=[]
    for cut in cuts:
        if cut[0]<min_range:
            continue
        re.append(cut[1:])
        crop_ax = (0, cut[1], w, cut[2])
        img_arr = np.array(pic.crop(crop_ax))#crop(左，上，右，下）
        imglist.append(img_arr)
    return imglist,re, True


def simple_cut(vert):
    begin, end = 0, 0
    cuts = []
    for i, count in enumerate(vert):
        if count >= min_thresh and begin == 0:
            begin = i
        elif count >= min_thresh and begin != 0:
            continue
        elif count <= min_thresh and begin != 0:
            end = i
            # print (begin, end), count
            if end - begin >= min_range:
                cuts.append((begin, end))

                begin = 0
                end = 0
                continue
        elif count <= min_thresh or begin == 0:
            continue
    return cuts


#pic_path = './2.JPG'
#save_path = './cutimg/'
save_size=(32,32)

def Cut(pic_path):
    src_pic = Image.open(pic_path).convert('L')#转为灰度图像
    src_arr = np.array(src_pic)
    src_img=cv2.imread(pic_path)#原图像
    #img_arr=cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)#灰度图像

    threshold = OTSU_enhance(src_arr)
    bin_arr = np.where(src_arr < threshold, 0, 255)  # 先用大律阈值将图片二值化处理

    horz = horizon(bin_arr)  # 获取到文本 水平 方向的投影
    line_list, cuts,flag = cut_line(horz, src_pic)  # 把文字（行）所在的位置切割下来
    if flag == False:  # 如果flag==False 说明没有切到一行文字
        exit()
    re=[]
    w,h=src_pic.size
    for i in range(len(line_list)):
        line_arr=line_list[i]
        cv2.line(src_img,(0,cuts[i][0]),(w,cuts[i][0]),(0,0,255))
        cv2.line(src_img, (0,cuts[i][1]), (w,cuts[i][1]), (0, 0,255))
        '''cv2.imshow("image of a line",line_arr)
        cv2.waitKey(0)'''
        line_arr = np.where(line_arr < threshold, 0, 255)
        line_img = Image.fromarray(line_arr.astype("uint8"))
        width, height = line_img.size

        vert = vertical(line_arr)  # 获取到该行的 垂直 方向的投影
        cut = simple_cut(vert)  # 直接对目标行进行切割

        for x in range(len(cut)):
            cv2.line(src_img, (cut[x][0],cuts[i][0]), (cut[x][0],cuts[i][1]), (0, 0,255))
            cv2.line(src_img, (cut[x][1],cuts[i][0]), (cut[x][1],cuts[i][1]), (0, 0,255))
            ax = (cut[x][0] - 1, 0, cut[x][1] + 1, height)
            temp = line_img.crop(ax)
            temp=temp.resize(save_size)
            temp=np.array(temp)
            '''cv2.imshow("the cutting charecter",temp)
            cv2.waitKey(0)'''
            temp = np.where(temp < threshold, 0, 255)
            re.append(temp)
    re=np.array(re)
    cv2.imshow("the cutting image",src_img)
    cv2.waitKey(0)
    return re