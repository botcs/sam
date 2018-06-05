import dlib
import numpy as np
import cv2
import glob
import os
import time
import datetime
import argparse
import shutil
import threading
from multiprocessing import Pool
import sys
sys.path.insert(0, '/home/csbotos/sam/')
from utils import AlignDlib

parser = argparse.ArgumentParser()

parser.add_argument('--list', required=True, help='Path to BUFFER directory')
parser.add_argument('--recheck', type=int, default=9999999,
                    help='Time interval (in seconds) between finishing and starting a next sweep')
parser.add_argument('--workers', type=int, default=1,
                    help='Number of parallel workers')
parser.add_argument('--size', type=int, default=96,
                    help='Size of the aligned face (in pixels)')
parser.add_argument('--dlibFacePredictor', type=str, help='Path to dlib\'s face predictor.',
                    default='/home/csbotos/sam/weights/shape_predictor_68_face_landmarks.dat')
parser.add_argument('--delete', action='store_true', help='Delete source images without faces')
args = parser.parse_args()
marker_tag = 'aligned'
print('Arguments passed:\n', args, '\n')
def isProcessed(fname):
    return fname.find(marker_tag) > -1


def mkstamp(date):
    parts = date.split('.')
    if len(parts) == 4:
        timestamp = int(parts[-2]) * 1000  + int(parts[-1][:3])
    else:
        dt = datetime.datetime.strptime(date, "%Y-%m-%d.%H-%M-%S.%f")
        timestamp = time.mktime(dt.timetuple()) + (dt.microsecond / 1000000.0)
    
        timestamp *= 1000
    
    return int(timestamp)

aligner = AlignDlib('weights/shape_predictor_68_face_landmarks.dat')
def mkdirMaybe(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        

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
    aligned_img = aligner.align(96, rgbImg)
    
    if aligned_img is None:

        if args.delete:
            print('[%6d / %6d] No face was found: REMOVE %30s' % (idx, total_imgs, img_path))
            os.remove(img_path)
        else:
            print('[%6d / %6d] No face was found: %30s' % (idx, total_imgs, img_path))
        return
    
    img_name = os.path.basename(img_path)
    try:
        mkdirMaybe(os.path.join(os.path.dirname(img_path), marker_tag))
    except FileExistsError as e:
        pass
    target_path = os.path.join(os.path.dirname(img_path), marker_tag, img_name)
    
    # CV2 need Blue Green Red ordering for writing
    bgr_aligned_img = cv2.cvtColor(aligned_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(target_path, bgr_aligned_img)
    print('[%6d / %6d] Extracted to: %30s' % (idx, total_imgs, target_path))

if __name__ == '__main__':
    print(args, flush=True)
    while args.recheck > 0:
        img_paths = open(args.list).read().splitlines()
        worker_args = [(i, path) for (i, path) in enumerate(img_paths) if not isProcessed(path)]
        total_imgs = len(worker_args)
        print('Found %7d files in total' % total_imgs)
        p = Pool(args.workers)
        start_time = time.time()
        p.map(procImg, worker_args)
        total_time = time.time() - start_time
        FPS = len(img_paths) / total_time
        print('Finished Aligning %d images in %d sec, FPS=%3.2f, waiting %d sec before next sweep' % (len(img_paths), total_time, FPS, args.recheck))
        time.sleep(args.recheck)