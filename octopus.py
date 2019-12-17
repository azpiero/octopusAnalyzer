import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
import glob
import csv
import math
import shutil

def pil2cv(image):
    new_image = np.array(image,dtype=np.uint8)
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = new_image[:,:,::-1]
    elif  new_image.shape[2] == 4:
        new_image = new_image[:,:,[2,1,0,3]]
    return new_image

def tiftoimg(img_pil,folder):
    img_length = img_pil.n_frames
    img_0 = ""
    for i in range(img_length):
        img_pil.seek(i)
        img = img_pil.copy()
        img_cv2 = pil2cv(img)
        if(i == 0):
            img_0 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        pil_img_ = Image.fromarray(img_rgb)
        pil_img_.save(os.path.join(folder , "./original/" , str(i) +  "_original.jpg"))
    return os.path.join(folder , "original/0_original.jpg")

def detect_pointed_chromophore(x,y):
    # 矩形用の画像をコピー
    shutil.rmtree(os.path.join(folder ,"pointed_area") )
    shutil.copytree(os.path.join(folder , "./original/") , os.path.join(folder ,"pointed_area") )
    flist = glob.glob(os.path.join(os.path.join(folder ,"pointed_area/*") ))
    print(flist)
    i = 0
    # TODO 順番の調整 
    # これが思い通りの順番とは限らなくない？
    for file in flist:
        os.rename(file, os.path.join(folder ,"pointed_area/" , str(i) + '_pointed_area.jpg'))
        
        img_pil = Image.open(os.path.join(folder ,"pointed_area/" , str(i) + '_pointed_area.jpg'))
        img_cv2 = pil2cv(img_pil)
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        img_rgb = morph_and_blur(img_rgb)
        # 2値化
        # ハイパーパラメータ
        # ret,img_thresh = cv2.threshold(img_rgb,130,255,cv2.THRESH_BINARY_INV)
        # TODO このthreshを適切に決めなければならない
        ret,img_thresh = cv2.threshold(img_rgb,130,255,cv2.THRESH_BINARY_INV)

        #contours,hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours,hierarchy = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 最小の距離を持つ輪郭の番号get
        min_distance = 10000
        center_of_gravity = 0
        for j in range(len(contours)):
            cnt = contours[j]
            M = cv2.moments(cnt)
            if M['m10'] == 0:
                continue
            distance = math.sqrt( ( M['m10']/M['m00'] - x )**2 + ( M['m01']/M['m00'] - y )**2 )
            if min_distance > distance:
                min_distance = distance
                center_of_gravity = j

        print("min_distance",min_distance)
        print("center_of_gravity",center_of_gravity)

        cv2.drawContours(img_rgb, contours, center_of_gravity, (255, 0, 0), 1)
        cv2.imwrite(os.path.join(folder ,"pointed_area/" , str(i) + '_pointed_area_after.jpg') ,img_rgb)

        # cv2.polylines(img, contours[i], True, (255, 255, 255), 5)
        # 範囲を指定したマスク作る
        # img_mask_yellow = cv2.inRange(img_rgb, rgb_lower_yellow, rgb_upper_yellow)
        # マスク画像と元々の画像でandとると残ってるとこだけ
        # result_yellow = cv2.bitwise_and(img_rgb,img_rgb,mask=img_mask_yellow)
        # culculate_each_area(result_yellow)

        i+=1


def motion(event):  
    if event.dblclick == 1:
        x = int(math.floor(event.xdata))
        y = int(math.floor(event.ydata))
        print(x,y)
        # この点を持って全てのoriginal画像から所望の範囲を捜査する
        detect_pointed_chromophore(x,y)
        plt.title("double click")

    elif event.button == 1:
        plt.title("left click")

    elif event.button == 3:
        plt.title("right click")

    plt.draw()

def morph_and_blur(img):
    kernel = np.ones((3, 3),np.uint8)
    m = cv2.GaussianBlur(img, (3, 3), 0)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=2)
    m = cv2.GaussianBlur(m, (5, 5), 0)
    return m

def main():
    global imgMain
    global folder 
    folder =  "./octpas/result/"
    img_pil = Image.open("./octpas/adjuststack.tif")
    # まずtifをimgに変換
    fig = tiftoimg(img_pil,folder)
    #plt.imshow(fig)
    
    # 上から被せる
    #imgOver = np.zeros((500,500,3), np.uint8)
    imgMain = mpimg.imread(fig)
    
    ax = plt.subplot(111)
    ax.imshow(imgMain, interpolation='nearest', alpha=1)
    #ax.imshow(imgOver, interpolation='nearest', alpha=0.6)

    #pntr = Painter(ax, imgOver)
    # pntr = Painter(ax, imgMain)
    plt.connect('button_press_event', motion)
    plt.title('Double Click on the image to decide Chromophore')
    plt.show()

# TODO 引数としてファイルのfullpath
if __name__ == '__main__':
    main()
