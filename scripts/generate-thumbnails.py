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
from utils import send_query

parser = argparse.ArgumentParser()

parser.add_argument('--list', required=True, help='Path to BUFFER directory')
parser.add_argument('--recheck', type=int, default=9999999,
                    help='Time interval (in seconds) between finishing and starting a next sweep')
parser.add_argument('--workers', type=int, default=1,
                    help='Number of parallel workers')

args = parser.parse_args()
marker_tag = 'thumbnail'

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


def updateSQL(path):
    img_name = os.path.basename(path)
    ts = mkstamp(img_name[img_name.find('2018-'):-5])
    
    SQL_QUERY ='INSERT INTO thumbnail(PATH, TIMESTAMP) VALUES("%s", %d); '%(path, ts)
    send_query(SQL_QUERY, verbose=False)

detector = dlib.get_frontal_face_detector()
start_time = time.time()

def extractFace(img, offset=20):
    faces = detector(img, 1)
    if len(faces) == 0:
        return None
    largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
    imin, imax = largest_face.top(), largest_face.bottom()
    jmin, jmax = largest_face.left(), largest_face.right()

    imin = max(imin-offset, 0)
    jmin = max(jmin-offset, 0)
    imax = min(imax+offset, img.shape[0])
    jmax = min(jmax+offset, img.shape[1])

    x = img[imin:imax, jmin:jmax]
    return x


def mkdirMaybe(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        

def procImg(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print('Corrupt image: rm %30s' % img_path)
        os.remove(img_path)
        return
    
    img = extractFace(img)
    if img is None:
        print('No face was found: %30s' % img_path)
        return
    
    img_name = os.path.basename(img_path)
    try:
        mkdirMaybe(os.path.join(os.path.dirname(img_path), marker_tag))
    except FileExistsError as e:
        pass
    target_path = os.path.join(os.path.dirname(img_path), marker_tag, img_name)
    t = threading.Thread(target=updateSQL, args=[target_path])
    t.setDaemon(True)
    t.start()
    cv2.imwrite(target_path, img)
    print('Extracted to: %80s'%target_path)

if __name__ == '__main__':
    print(args, flush=True)
    while args.recheck > 0:
        img_paths = open(args.list).read().splitlines()
        #img_paths = [path for path in img_paths if not isProcessed(path)]
        print('Found %7d files in total' % len(img_paths))
        p = Pool(args.workers)
        start_time = time.time()
        list(p.map(procImg, img_paths))
        total_time = time.time() - start_time
        FPS = len(img_paths) / total_time
        print('Finished extraction in %d sec, FPS=%3.2f, waiting %d sec before next sweep' % (total_time, FPS, args.recheck))
        time.sleep(args.recheck)
