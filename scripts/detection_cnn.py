#! /usr/bin/env python3
# coding: utf-8


import face_recognition
import cv2
import torch
import torch.utils.data
import threading
import multiprocessing
import matplotlib.pyplot as plt

import os, time, datetime
from PIL import Image




all_paths = open('/home/csbotos/video/all_img.txt').read().splitlines()




class customDataset(torch.utils.data.Dataset):
    def __init__(self, paths, transform=None):
        super(customDataset, self).__init__()
        self.paths = paths
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        bgrImg = cv2.imread(self.paths[idx])
        img = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img)
        
        return img
    
dataset = customDataset(paths=all_paths)
dataloader = torch.utils.data.DataLoader(
    dataset, shuffle=False, batch_size=128, num_workers=4, collate_fn=lambda x: x
)




def path2ts(path):
    img_name = os.path.basename(path)
    date = img_name[img_name.find('2018-'):-5]
    dt = datetime.datetime.strptime(date, "%Y-%m-%d.%H-%M-%S.%f")
    timestamp = time.mktime(dt.timetuple()) + (dt.microsecond / 1000000.0)
    timestamp *= 1000
    return int(timestamp)


def mkdirMaybe(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)




def parallelProcessFrame(frame, locations, path):
    for top, right, bottom, left in locations:
        filename = "ts%d-%d-%d-%d-%d.jpg" % (path2ts(path),left, top, right, bottom)
        dirname = os.path.join(os.path.dirname(path), 'thumbnail')
        offset = int((right-left)/2 * 1.5)
        top = max(top-offset, 0)
        left = max(left-offset, 0)
        bottom = bottom+offset
        right = right+offset
        thumbnail = frame[top:bottom, left:right]
        thumbnail_path = os.path.join(dirname, filename)
        #display(thumbnail_path, Image.fromarray(thumbnail[:, :]))
        #print('->', thumbnail_path)
        cv2.imwrite(thumbnail_path, thumbnail[:, :, ::-1])
            


for batch_idx, frames in enumerate(dataloader):
    print('[%7d | %7d] ' % (min((batch_idx+1) * len(frames), len(dataset)), len(dataset)), 
          flush=True, end='')
    start_time = time.time()
    batch_locations = face_recognition.batch_face_locations(
        frames, number_of_times_to_upsample=0)
    
    paths = dataset.paths[batch_idx * len(frames) : (batch_idx+1) * len(frames)]
    for args in zip(frames, batch_locations, paths):
        t = threading.Thread(target=parallelProcessFrame, args=args, daemon=True)
        t.start()
        
    batch_time = time.time() - start_time
    FPS = dataloader.batch_size / batch_time
    print('%1.4f secs, FPS=%2.2f'%(batch_time, FPS))
    

