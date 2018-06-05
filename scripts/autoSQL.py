
# coding: utf-8

# In[3]:


from utils.sqlrequest import db_query
from glob import glob
import re
import datetime, time
import os
import shutil


# In[4]:


def ensureParentDirExists(path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        
def get_ts(path):
    # Format: <unix epoch time in msec>-<xmin>-<ymin>-<xmax>-<ymax>
    img_name = os.path.basename(path)
    ts = int(img_name.split('-')[0])
    return ts


# In[5]:


def moveToProcessed(tag):
    sourceRoot = os.path.join('/home/csbotos/video/unprocessed-recordings/', tag)
    targetRoot = os.path.join('/home/csbotos/video/auto-processed/', tag)
    sourcePaths = glob(os.path.join(sourceRoot, '**/*.jpg'), recursive=True)
    targetPaths = [re.sub(sourceRoot, targetRoot, sp) for sp in sourcePaths]
    for src, dst in zip(sourcePaths, targetPaths):
        ensureParentDirExists(dst)
        shutil.move(src, dst)
    return targetPaths


# In[6]:


fullframePaths = moveToProcessed('recordings')
thumbnailPaths = moveToProcessed('thumbnail')
alignedPaths = moveToProcessed('aligned')


# In[17]:


def filelist(tag):
    sourceRoot = os.path.join('/home/csbotos/video/auto-processed/', tag)
    targetRoot = os.path.join('/home/csbotos/video/auto-processed/', tag)
    sourcePaths = glob(os.path.join(sourceRoot, '**/*.jpg'), recursive=True)
    targetPaths = [re.sub(sourceRoot, targetRoot, sp) for sp in sourcePaths]
    return targetPaths


# In[18]:


fullframePaths = filelist('recordings')
thumbnailPaths = filelist('thumbnail')
alignedPaths = filelist('aligned')


# In[19]:


len(fullframePaths)


# In[21]:


# def insertToPhotosTable(paths): helyett
data = [(path,get_ts(path)) for path in fullframePaths]
db_query('INSERT IGNORE INTO photo(path, timestamp) VALUES(%s, %s)',data)


# In[35]:


photo_info = [re.split(r'[-]+', os.path.basename(fname).split('.')[0]) for fname in alignedPaths]
thumbnailPaths_set = set(thumbnailPaths)
data = []
for i,aligned_path in enumerate(alignedPaths):
    thumbnail_path = aligned_path.replace('aligned','thumbnail')
    if thumbnail_path in thumbnailPaths_set:
        data.append({
            'thumbnail_path': thumbnail_path,
            'aligned_path': aligned_path,
            'timestamp': photo_info[i][0],
            'xmin': photo_info[i][1],
            'ymin': photo_info[i][2],
            'xmax': photo_info[i][3],
            'ymax': photo_info[i][4],
        })


# In[36]:


print(data[0])
len(data),len(alignedPaths)


# In[37]:


db_query('''
INSERT IGNORE INTO thumbnail (photo_ID,thumbnail_path,aligned_path,xmin,ymin,xmax,ymax)
SELECT photo_ID,%(thumbnail_path)s,%(aligned_path)s,%(xmin)s,%(ymin)s,%(xmax)s,%(ymax)s
FROM photo
WHERE timestamp = %(timestamp)s
''',data)

