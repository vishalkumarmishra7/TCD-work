import cv2 
import scipy
from matplotlib import pyplot as plt
import skimage
from skimage import io
import os
import scipy.ndimage
import imageio
import numpy as np
import sys
from PIL import Image
from scipy import ndimage
import scipy.misc
import pandas as pd

error_imgs=pd.Series()
src = 'overflow/dissertation/nsfw data downloader/image_data'
# src = 'real_landscapes'
dest_img = 'overflow/dissertation/nsfw data downloader/image_data_GAN'

if not os.path.exists(os.path.join(dest_img,'real')):
    os.makedirs(os.path.join(dest_img,'real'))
if not os.path.exists(os.path.join(dest_img,'sketch')):
    os.makedirs(os.path.join(dest_img,'sketch'))
if not os.path.exists(os.path.join(dest_img,'join_train')):
    os.makedirs(os.path.join(dest_img,'join_train'))


def dodge(front,back):
    result=front*255/(255-back) 
    result[result>255]=255
    result[back==255]=255
    return result.astype('uint8')

def grayscale(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

blur_sigma=5


# def getListOfFiles(dirName):
#     # create a list of file and sub directories 
#     # names in the given directory 
#     listOfFile = os.listdir(dirName)
#     allFiles = list()
#     # Iterate over all the entries
#     for entry in listOfFile:
#         # Create full path
#         fullPath = os.path.join(dirName, entry)
#         # If entry is a directory then get the list of files in this directory 
#         if os.path.isdir(fullPath):
#             allFiles = allFiles + getListOfFiles(fullPath)
#         else:
#             allFiles.append(fullPath)
                
#     return allFiles

# listOfFiles = getListOfFiles(src)

# for i,image_name in enumerate(listOfFiles):
#     print (image_name)

i=0
for root, dirs, files in os.walk(src):
    for name in files:  
        
        if(i%1000)==0:
            print (i, "images processed")
        i+=1
        
        image_name=os.path.join(root, name)
#         print(os.path.join(root, name)) 

        try:

            s = imageio.imread(image_name)
            s = cv2.resize(s, (500,500), interpolation = cv2.INTER_AREA)

            type_of_img=os.path.basename(os.path.dirname(image_name))
            nm=os.path.basename(image_name)
            full_nm_real=os.path.join(dest_img,'real',type_of_img+'_'+nm[:-4]+'.jpg')
            plt.imsave(full_nm_real, s)

            g=grayscale(s)
            g = ndimage.gaussian_filter(g, sigma=blur_sigma)
            ii = 255-g
            b = scipy.ndimage.filters.gaussian_filter(ii,sigma=10)
            r= dodge(b,g)

            full_nm_sketch=os.path.join(dest_img,'sketch',type_of_img+'_'+nm[:-4]+'.jpg')
            plt.imsave(full_nm_sketch, r, cmap='gray', vmin=0, vmax=255)

        #join

            images = [Image.open(x) for x in [full_nm_real,full_nm_sketch]]
            widths, heights = zip(*(i.size for i in images))
            total_width = sum(widths)
            max_height = max(heights)
            new_im = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            for im in images:
                new_im.paste(im, (x_offset,0))
                x_offset += im.size[0]


            full_nm_join=os.path.join(dest_img,'join_train',type_of_img+'_'+nm[:-4]+'.jpg')
            new_im.save(full_nm_join)
        #     plt.imsave(os.path.join(dest_img,str(i)), r, cmap='gray', vmin=0, vmax=255)

        except:
            error_imgs=error_imgs.append(pd.Series(full_nm_real))
            print('error')

error_imgs.to_csv('error_images.csv')