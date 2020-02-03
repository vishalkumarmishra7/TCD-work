#!/usr/bin/env python
# coding: utf-8

# ### Initialising

# In[69]:


import os
import requests
import pandas as pd
### Intitialise the download process
thisdir = "C:/Users/adhis/Desktop/git_rep/nsfw_data_source_urls-master"
destination = "test"
completed_url_path = "completed_urls.csv"

if os.path.isdir(destination) == False:
    os.makedirs(destination)

try:
    pd.read_csv(completed_url_path)
except:
    df = pd.DataFrame(columns = ['URLs'])


# ### Listing all url files

# In[26]:


f_list = []
for r, d, f in os.walk(thisdir):
    for file in f:
        if ".txt" in file:
            f_list.append(os.path.join(r, file))


# ### Downloading the images with classification labels as names

# In[67]:


p_urls = []
c_urls = []
for i in [f_list[0]]:
    u_list = list(open(i))
    cnt = 0
    for j in u_list[11:20]:
        j.replace('\n','')
        cnt += 1
        if i.split('\\')[-2] != 'nsfw_data_source_urls-master':
            file_name = i.split('\\')[-2] + '_' + str(cnt)
            if cnt % 100 == 0:
                print('URLs processed: ',cnt)
            try:
                img_data = requests.get(j).content
                with open(destination+'/'+file_name+'.jpg', 'wb') as handler:
                    handler.write(img_data)
            except:
                p_urls.append(j)
        c_urls.append(j)

df = df.append(pd.DataFrame(c_urls, columns = ['URLs']))
df.to_csv(completed_url_path,index = False)

