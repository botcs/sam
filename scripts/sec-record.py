import numpy as np
import datetime
import threading
import time
import cv2
import os
import argparse
import sys

#cap0 = cv2.VideoCapture(1)
#cap1 = cv2.VideoCapture(2)

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--hd', action='store_true')
parser.add_argument('--path', default='recordings')
parser.add_argument('--cam', nargs='+', type=int, default=[0])
args = parser.parse_args()



def isPressed(char, input):
    return ord(char) == input


def TimeStamp():
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d.%H-%M-%S.') + str(ts)


def Record(recordDir, captureDevice):
    start = time.time()
    if not os.path.exists(recordDir):
        os.makedirs(recordDir)
        print('New directory: %40s' % recordDir)

    try:
        while True:
            ret, frame = captureDevice.read()
            currTS = TimeStamp()
            filename = str(currTS) + '.jpg'
            path = os.path.join(recordDir, filename)
            if args.debug:
                print('DEBUG  capture succesful:', ret, ' shape:', frame.shape)
                print('DEBUG  Saved to: %40s' % path)
                cv2.imshow(recordDir, frame)
                cv2.waitKey(1)
            else:
                # Sometimes cv2.imwrite stucks for a while, but the loop must go on
                threading.Thread(target=cv2.imwrite, args=(path, frame)).start()
    except KeyboardInterrupt:
        print('Manually interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)



caps = list(map(cv2.VideoCapture, args.cam))
if args.hd:
    for i in range(len(caps)):
        caps[i].set(3, 1280)
        caps[i].set(4, 720)

for cap, cam_id in zip(caps, args.cam):
    if args.debug:
        print('DEBUG ', cap)

    camDir = os.path.join(args.path, '%02d' % cam_id)
    Record(camDir, cap)
    #threading.Thread(target=Record, args=(camDir, cap))




# When everything done, release the capture
#cap0.release()
#cap1.release()
