
from PIL import Image
import os
from multiprocessing import Pool
import random
import datetime

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

# Running pooled processes
pool = Pool()
pool.map(resize , l)

# Getting porn images
src = '/users/pgrad/amittal/overflow/dissertation/nsfw data downloader/image_data/'
typ = '1' # For Porn image

# Creating list of files in the folder
l2 = os.listdir(src)

l3 = []

for i in l2:
    if i[:-3] == 'jpg':
        l3.append(i)    

l3 = random.sample(l3,len(l))

# Running pooled processes
pool = Pool()
pool.map(resize , l3)
