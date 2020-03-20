from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import random

import os
import time
import cv2
import numpy as np
from shutil import copyfile

from matplotlib import pyplot as plt
from IPython import display


nsfw_path = 'overflow/dissertation/nsfw data downloader/image_data'
l = os.listdir(nsfw_path)

cnt = 0
for i in l:
    cnt += 1
    if cnt > 10000:
        break
    ind = random.randint(1,len(l))
    if cnt%3 == 0:
        if cnt%2 == 0:
            tt = 'val'
        else:
            tt = 'train'
    else:
        tt = 'test'
    copyfile(os.path.join(nsfw_path, l[ind]), 'nsfw/' + tt + '/' + l[ind])

PATH = 'nsfw/train'