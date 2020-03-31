
from PIL import Image
import os
from multiprocessing import Pool
import random
import datetime
import shutil

# defining function to resize
def resize(name):
    try:
        im = Image.open(src + name)
        out = im.resize((100, 100))
        r = random.randint(1,10)
        if r <= 4:
            if r == 1:
                subfolder = 'val'
            else:
                subfolder = 'test'
        else:
            subfolder = 'train'
        out.save(dest + subfolder + '/' + typ + '_' + name)
    except:
        print('Issue with ',name)
    return datetime.datetime.now()
    
dest = '/notbackedup/studentstorage/users/pgrad/amittal/dissertation/Data/' # For destination folder
src = '/notbackedup/studentstorage/users/pgrad/amittal/dissertation/Data/non_porn_source/' # For non porn images source folder 

typ = '0' # For non porn images

# Creating list of files in the folder
l = os.listdir(src)
l = [i for i in l if (i[-4:].lower() == 'jpeg') or (i[-3:].lower() == 'jpg')]
print('Number of images in non porn = ',len(l))

# Deleting previous runs and creating fresh folders
shutil.rmtree(dest + 'test')
shutil.rmtree(dest + 'train')
shutil.rmtree(dest + 'val')
os.mkdir(dest + 'test')
os.mkdir(dest + 'train')
os.mkdir(dest + 'val')

# Running pooled processes
pool = Pool()
pool.map(resize , l)

# Getting porn images
src = '/users/pgrad/amittal/overflow/dissertation/nsfw data downloader/image_data/'
typ = '1' # For Porn image

# Creating list of files in the folder
l2 = os.listdir(src)
l2 = [i for i in l2 if (i[-4:].lower() == 'jpeg') or (i[-3:].lower() == 'jpg')]
l2 = random.sample(l2,len(l))
print('Number of images in non porn = ',len(l2))

# Running pooled processes
pool = Pool()
pool.map(resize , l2)
