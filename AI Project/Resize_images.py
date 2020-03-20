from PIL import Image
import os, sys
from random import randint

def resize(path):
    dirs = os.listdir( path )
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((200,200), Image.ANTIALIAS)
            if(randint(1,10) <= 3):
                imResize.save(path[:-1] +'_test/'+ item + '_resized.jpg', 'JPEG', quality=90)
            else:
                imResize.save(path[:-1] + '_train/    ' + item + '_resized.jpg', 'JPEG', quality=90)

resize(path = "GAN/real/real_landscapes/")
resize(path = "GAN/real/painted_landscapes/")