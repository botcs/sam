import sys
sys.path.insert(0, '/home/csbotos/sam/')

from utils import TripletImageLoader
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
import dlib
import numpy as np
from utils import prepareOpenFace
from utils import send_query, send_large_query

import time
import datetime
from IPython.display import display
from IPython.display import Image as im
from PIL import Image


# In[2]:

print('-'*80)
print('AutoEmbedding: start time\t', datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d.%H-%M-%S'))

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
cudnn.benchmark = True


# ## Download complete, sorted path database

# In[3]:


query_result = send_large_query('SELECT aligned_path FROM thumbnail ORDER BY thumbnail_ID', 
                                batch_size=50000, verbose=False)
database_paths = [q['aligned_path'] for q in query_result]


# ## Define a batch image loader that _FETCHES_ the pre-aligned faces

# In[4]:


class customDataset(torch.utils.data.Dataset):
    def __init__(self, paths, transform=None):
        super(customDataset, self).__init__()
        self.paths = paths
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        if self.transform is not None:
            img = self.transform(img)
        
        return img
    
dataset = customDataset(
    paths=database_paths,
    transform=transforms.Compose([
        transforms.Resize(96),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
    ]), 
)
dataloader = torch.utils.data.DataLoader(
    dataset, 
    shuffle=False, 
    batch_size=1024, 
    num_workers=8
)


# ## Load the face-embedding network
# ### make sure that it will be optimized for inference

# In[5]:


net = prepareOpenFace()
net.load_state_dict(torch.load('/home/csbotos/sam/weights/openface.pth'))
net = net.eval()
net.cuda()
for p in net.parameters():
    p.requires_grad = False
cudnn.benchmark = True


# In[ ]:


embs = None
for batch_idx, imgs in enumerate(dataloader, 1):
    #torch.cuda.empty_cache()
    X = Variable(imgs, volatile=True, requires_grad=False).cuda()
    if embs is None:
        embs = net(X)[0]
    else:
        embs = torch.cat([embs, net(X)[0]])
    print('[%5d|%5d]'%(batch_idx, len(dataloader)))
    


embedding_database = {
    'paths': dataset.paths,
    'embeddings': embs.cpu()
}

torch.save(embedding_database, 'AUTO_EMBEDDING_DATABASE.tar')


# In[11]:


print('Size:', len(embs))

print('AutoEmbedding: done.\t', datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d.%H-%M-%S'))
print('='*80)
