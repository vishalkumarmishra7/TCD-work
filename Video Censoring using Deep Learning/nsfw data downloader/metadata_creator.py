import os
import pandas as pd

stats = []

path = "overflow/dissertation/nsfw data downloader/image_data/"

for i in os.listdir(path):
    if(i[0] != '.'):
        stats.append(list(os.stat(path+'/'+i)).append(i));

df = pd.DataFrame(stats)

df.columns = ['mode', 'ino', 'dev', 'nlink', 'uid', 'gid', 'size', 'atime', 'mtime', 'ctime', 'filename']

df.to_csv("overflow/dissertation/nsfw data downloader/metadata_file.csv")