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
parser.add_argument('--dst', required=True, help='Path to STORE directory')
parser.add_argument('--nodryrun', action='store_true')
parser.add_argument('--workers', type=int, default=1,
                    help='Number of parallel workers')
args = parser.parse_args()


detector = dlib.get_frontal_face_detector()
start_time = time.time()
# counter = 0
# for counter, img_path in enumerate(img_paths, 1):
def procImg(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(
            #'x[%7d / %7d]' % (counter, len(img_paths)),
            #'FPS=%3.2f' % FPS,
            'Corrupt image: rm %30s' % (img_path),
            sep='   '
        )

        if args.nodryrun:
            os.remove(img_path)
        return
    img_name = os.path.basename(img_path)
    target_path = os.path.join(args.dst, img_path)
    rects = detector(img)
    # FPS = counter / (time.time() - start_time)
    if len(rects) > 0:
        print(
            #'+[%7d / %7d]' % (counter, len(img_paths)),
            #'FPS=%3.2f' % FPS,
            '%2d face(s) detected: %30s -> %30s' % (len(rects), img_path, target_path),
            sep='   '
        )
        if args.nodryrun:
            shutil.move(img_path, target_path)
    else:
        print(
            #'-[%7d / %7d]' % (counter, len(img_paths)),
            #'FPS=%3.2f' % FPS,
            'No face detected: rm %30s' % (img_path),
            sep='   '
        )
        if args.nodryrun:
            os.remove(img_path)


if __name__ == '__main__':
    img_paths = glob.glob(args.src + '/**/*.jpg', recursive=True)
    p = Pool(args.workers)
    print(p.map(procImg, img_paths))
