import dlib
import numpy as np
import cv2
import glob
import os
import time
import argparse
import shutil

from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument('--src', required=True, help='Path to BUFFER directory')
parser.add_argument('--dst', help='Path to STORE directory')
parser.add_argument('--recheck', type=int, default=0,
                    help='Time interval (in seconds) between finishing and starting a next sweep')
parser.add_argument('--nodryrun', action='store_true')
parser.add_argument('--workers', type=int, default=1,
                    help='Number of parallel workers')
args = parser.parse_args()

markerTag = 'hasface-'
def isProcessed(fname):
    return fname.find(markerTag) > -1

detector = dlib.get_frontal_face_detector()
start_time = time.time()

def procImg(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print('Corrupt image: rm %30s' % (img_path))
        if args.nodryrun:
            os.remove(img_path)
        return
    img_name = os.path.basename(img_path)
    if args.dst is not None:
        target_path = os.path.join(args.dst, markerTag + img_name)
    else:
        # easiest solution for avoiding multiple process of the same image
        target_path = os.path.join(os.path.dirname(img_path), markerTag + img_name)
    rects = detector(img)
    if len(rects) > 0:
        print('%2d face(s) detected: %30s -> %30s' % (len(rects), img_path, target_path))
        if args.nodryrun:
            shutil.move(img_path, target_path)
    else:
        print('No face detected: rm %30s' % (img_path))
        if args.nodryrun:
            os.remove(img_path)


if __name__ == '__main__':
    print(args, flush=True)
    while args.recheck > 0:
        img_paths = glob.glob(args.src + '/**/*.jpg', recursive=True)
        img_paths = [path for path in img_paths if isProcessed(path)]
        print('Found %7d files in total' % len(img_paths))
        p = Pool(args.workers)
        start_time = time.time()
        list(p.map(procImg, img_paths))
        total_time = time.time() - start_time
        FPS = len(img_paths) / total_time
        print('Finished sweep in %d sec, FPS=%3.2f, waiting %d sec before next sweep' % (total_time, FPS, args.recheck))
        time.sleep(args.recheck)
