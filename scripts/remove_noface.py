import dlib
import numpy as np
import cv2
import glob
import os
import time
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--src', required=True, help='Path to buffer directory')
parser.add_argument('--dst', required=True, help='Path to store directory')
parser.add_argument('--nodryrun', action='store_true')
args = parser.parse_args()


detector = dlib.get_frontal_face_detector()
img_paths = glob.glob(args.src + '/**/*.jpg', recursive=True)
start_time = time.time()
for counter, img_path in enumerate(img_paths, 1):
    # Capture frame-by-frame
    img = cv2.imread(img_path)
    if img is None:
        print(
            'x[%7d / %7d]' % (counter, len(img_paths)),
            'FPS=%3.2f' % FPS,
            'Corrupt image: rm %30s' % (img_path),
            sep='   '
        )

        if args.nodryrun:
            os.remove(img_path)
        continue
    img_name = os.path.basename(img_path)
    target_path = os.path.join(args.dst, img_path)
    rects = detector(img)
    FPS = counter / (time.time() - start_time)
    if len(rects) > 0:
        print(
            '+[%7d / %7d]' % (counter, len(img_paths)),
            'FPS=%3.2f' % FPS,
            '%2d face(s) detected: %30s -> %30s' % (len(rects), img_path, target_path),
            sep='   '
        )
        if args.nodryrun:
            shutil.move(img_path, target_path)
    else:
        print(
            '-[%7d / %7d]' % (counter, len(img_paths)),
            'FPS=%3.2f' % FPS,
            'No face detected: rm %30s' % (img_path),
            sep='   '
        )
        if args.nodryrun:
            os.remove(img_path)
