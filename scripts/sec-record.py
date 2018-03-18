import pdb
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


def mkdirMaybe(recordDir):
    if not os.path.exists(recordDir):
        os.makedirs(recordDir)
        print('New directory: %40s' % recordDir)


def isDeviceWorking(captureDevice):
    # returns False if not able to read a frame
    return captureDevice.read()[0]

class Record(threading.Thread):
    def __init__(self, camID):
        super(Record, self).__init__()
        self._isRunning = False
        self.camID = camID

    def capture(self):
        self.captureDevice = cv2.VideoCapture(camID)
        if args.hd:
            self.captureDevice.set(3, 1280)
            self.captureDevice.set(4, 720)

        self.recordDir = os.path.join(args.path, 'camdir-%d' % self.camID)
        self.available = isDeviceWorking(self.captureDevice)
        if not self.available:
            print('WARNING: failed to capture device %d... NOT RECORDING'%camID)
            return False
        print('Recording device %d started, target path:%40s'%(camID, self.recordDir))
        mkdirMaybe(self.recordDir)
        return True

    def run(self):
        succesful = self.capture()
        if not succesful:
            return

        self._isRunning = True
        startTime = time.time()
        count = 0
        while self._isRunning:
            ret, frame = self.captureDevice.read()
            count += 1
            currTS = TimeStamp()
            filename = str(currTS) + '.jpg'
            path = os.path.join(self.recordDir, filename)
            if args.debug:
                print('DEBUG  capture succesful:', ret, ' shape:', frame.shape)
                print('DEBUG  Saved to: %40s' % path)
                #cv2.imshow(self.recordDir, frame)
                #cv2.waitKey(1)
            else:
                # Sometimes cv2.imwrite stucks for a while, but the loop must go on
                threading.Thread(target=cv2.imwrite, args=(path, frame)).start()
        totalTime = time.time() - startTime
        print('Finishing Recording thread, releasing device:', self.camID,
              'frame count: %7d\t'%count,
              'time  (sec): %7d\t'%int(totalTime),
              '        FPS: %2.2f'%(count / totalTime))
        self.captureDevice.release()

    def stop(self):
        self._isRunning = False


threads = []
print('Starting Recording threads...')
for camID in args.cam:

    camDir = os.path.join(args.path, 'camdir-%02d' % camID)
    thread = Record(camID)
    thread.start()
    threads.append(thread)

while threading.activeCount() > 0:
    try:
        time.sleep(1.)
    except KeyboardInterrupt:
        print('Manually interruped')
        print('Stopping Recording threads...')
        for thread in threads:
            thread.stop()
        break




# When everything done, release the capture
#cap0.release()
#cap1.release()
