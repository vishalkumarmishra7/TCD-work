import os
import numpy as np
from PIL import Image
import cv2

l = os.listdir("C:\\Users\\adhis\\Desktop\\TCD Stuff\\Scalable Computing\\Captcha Project 3\\test")

s = np.random.uniform(low = 1, high = len(l), size = round(len(l)*0.3))
s = [int(i) for i in s]

val = [l[i] for i in s]
train = list(set(l) - set(val))

for i in val:
    img = cv2.imread('C:\\Users\\adhis\\Desktop\\TCD Stuff\\Scalable Computing\\Captcha Project 3\\test\\'+i,1)
    # img = cv2.medianBlur(img, 5)
    # img = cv2.Canny(img, 100, 200)
    cv2.imwrite('C:\\Users\\adhis\\Desktop\\TCD Stuff\\Scalable Computing\\Captcha Project 3\\val\\'+i,img)
    print(i+' Done')

for i in train:
    img = cv2.imread('C:\\Users\\adhis\\Desktop\\TCD Stuff\\Scalable Computing\\Captcha Project 3\\test\\'+i,1)
    # img = cv2.medianBlur(img, 5)
    # img = cv2.Canny(img, 100, 200)
    cv2.imwrite('C:\\Users\\adhis\\Desktop\\TCD Stuff\\Scalable Computing\\Captcha Project 3\\train\\'+i,img)
    print(i+' Done')
