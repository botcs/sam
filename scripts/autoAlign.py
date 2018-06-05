import dlib
import numpy as np
import cv2
import glob
import os
import time
import datetime
import argparse
import shutil
import re
import threading
from multiprocessing import Pool
import sys
sys.path.insert(0, '/home/csbotos/sam/')
from utils import AlignDlib


print('-'*80)
print('AutoAlign: start time\t', datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d.%H-%M-%S'))


parser = argparse.ArgumentParser()

parser.add_argument('--workers', type=int, default=1,
                    help='Number of parallel workers')
parser.add_argument('--size', type=int, default=96,
                    help='Size of the aligned face (in pixels)')
parser.add_argument('--upsample', type=int, default=0,
                    help='Number of upsamples befor HAAR detection (Dlib)')
parser.add_argument('--dlibFacePredictor', type=str, help='Path to dlib\'s face predictor.',
                    default='/home/csbotos/sam/weights/shape_predictor_68_face_landmarks.dat')
parser.add_argument('--root', type=str, help='Root of unprocessed images',
                    required=True)
args = parser.parse_args()



fullframe_tag = 'recordings'
thumbnail_tag = 'thumbnail'
aligned_tag = 'aligned'


print('Arguments passed:\n', args, '\n')


print('Setting working directory...')
print('  from:', os.getcwd())
os.chdir(args.root)
print('  to:  ', os.getcwd())

def mkstamp(date):
    parts = date.split('.')
    if len(parts) == 4:
        timestamp = int(parts[-2]) * 1000  + int(parts[-1][:3])
    else:
        dt = datetime.datetime.strptime(date, "%Y-%m-%d.%H-%M-%S.%f")
        timestamp = time.mktime(dt.timetuple()) + (dt.microsecond / 1000000.0)
    
        timestamp *= 1000
    
    return int(timestamp)

aligner = AlignDlib(args.dlibFacePredictor)
def ensureParentDirExists(path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def parallelWrapper(arg):
    global checklist
    try:
        procImg(arg)
    except KeyboardInterrupt:
        pass

def path2ts(path):
    # Format: <unix epoch time in msec>-<xmin>-<ymin>-<xmax>-<ymax>
    img_name = os.path.basename(path)
    ts = int(img_name.split('-')[0])
    return ts

def procImg(arg):
    idx, img_path = arg
    bgrImg = cv2.imread(img_path)

    if bgrImg is None:
        print('[%6d / %6d] Corrupt image: REMOVE %30s' % (idx, total_imgs, img_path))
        try:
            os.remove(img_path)
        except FileNotFoundError as e:
            pass
        return
    
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    bbs = aligner.getAllFaceBoundingBoxes(bgrImg, dlib_upsample=args.upsample)
    
    # WARNING!!
    # we assume every processed image has at least one face... 
    # because only those are saved, therefore we should extract it from filename
    # instead of simply just discarding it
    if len(bbs) == 0:
        fname = os.path.splitext(os.path.basename(img_path))[0]
        ts, left, top, right, bottom = map(int, fname.split('-'))
        dlib.rectangles([dlib.rectangle(left, top, right, bottom)])
        
    multialign = aligner.multiAlign(args.size, rgbImg, bbs=bbs)
    if multialign is None:
        print('[%6d / %6d] No face was found: %30s' % (idx, total_imgs, img_path))
        return
    

    thumbnails, aligned_imgs = multialign 
    dirname = os.path.dirname(os.path.abspath(img_path))
    
    for bb, thumbnail, aligned_img in zip(bbs, thumbnails, aligned_imgs):
        timestamp = path2ts(img_path)
        img_name = '%d-%d-%d-%d-%d.jpg' % (timestamp, bb.left(), bb.top(), bb.right(), bb.bottom())
        
        thumbnail_path = re.sub(fullframe_tag, thumbnail_tag, img_path)
        aligned_path = re.sub(fullframe_tag, aligned_tag, img_path)
        
        ensureParentDirExists(thumbnail_path)
        cv2.imwrite(thumbnail_path, thumbnail[:, :, ::-1])
        print('[%6d / %6d] save thumbnail: %30s' % (idx, total_imgs, os.path.abspath(thumbnail_path)))
   

        ensureParentDirExists(aligned_path)
        cv2.imwrite(aligned_path, aligned_img[:, :, ::-1])
        print('[%6d / %6d] save aligned:   %30s' % (idx, total_imgs, os.path.abspath(aligned_path)))
        

if __name__ == '__main__':
    img_paths = glob.glob(os.path.join(fullframe_tag, '**/*.jpg'), recursive=True)
    worker_args = [(i, path) for (i, path) in enumerate(img_paths)]
    total_imgs = len(worker_args)
    #TODO: Do send - recv between threads to count
    checklist = [False for _ in range(total_imgs)]
    print('Found %7d files in total' % total_imgs)
    p = Pool(args.workers)
    start_time = time.time()
    try:
        p.map(parallelWrapper, worker_args)
        total_time = time.time() - start_time
        FPS = total_imgs / total_time
        print('Finished Aligning %d images in %d sec, FPS=%3.2f' % (total_imgs, total_time, FPS))
    except KeyboardInterrupt:
        print('\n\nInterrupted manually, terminating workers...\n\n')
        total_time = time.time() - start_time
        completed_imgs = sum(checklist)
        FPS = completed_imgs / total_time
        print('Finished Aligning %d images in %d sec, FPS=%3.2f' % (completed_imgs, total_time, FPS))
 
 
 
print('AutoAlign: done.\t', datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d.%H-%M-%S'))
print('='*80)
