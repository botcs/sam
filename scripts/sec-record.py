import pdb
import numpy as np
import datetime
import threading
import time
import cv2
import os
import argparse
import sys
import dlib
import glob


parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--hd', action='store_true')
#parser.add_argument('--face', action='store_true')
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


def saveMaybe(path, frame, detector):
    # Save frame only if dlib recognizes a face on it.
    # Do it in grayscale to save some compute
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hasFace = len(detector(gray, 0)) > 0
    if hasFace:
        cv2.imwrite(path, frame)
        print('DEBUG  Saved to: %40s' % path)
    return hasFace

class Record(threading.Thread):
    def __init__(self, camID):
        super(Record, self).__init__()
        self.detector = dlib.get_frontal_face_detector()
        self._isRunning = False
        self.camID = camID
        self.count = 0

    def capture(self):
        self.captureDevice = cv2.VideoCapture(camID)
        if args.hd:
            self.captureDevice.set(3, 1280)
            self.captureDevice.set(4, 720)

        self.recordDir = os.path.join(args.path, 'camdir-%d' % self.camID)
        self.available = isDeviceWorking(self.captureDevice)
        if not self.available:
            print('WARNING: failed to capture device %d... NOT RECORDING'%camID)
            print('write "ls /dev/video*" to list available devices')
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
        while self._isRunning:
            ret, frame = self.captureDevice.read()
            currTS = TimeStamp()
            filename = str(currTS) + '.jpg'
            path = os.path.join(self.recordDir, filename)
            if args.debug:
                print('DEBUG  capture succesful:', ret, ' shape:', frame.shape)
                #cv2.imshow(self.recordDir, frame)
                #cv2.waitKey(1)

            # if args.face:
            #     threading.Thread(
            #         target=saveMaybe, args=(path, frame, self.detector, self.count)
            #     ).start()
            #
            # else:
            # Sometimes cv2.imwrite stucks for a while, but the loop must go on
            threading.Thread(target=cv2.imwrite, args=(path, frame)).start()
            print('DEBUG  Saved to: %40s' % path)
            counter += 1

        totalTime = time.time() - startTime
        print('Finishing Recording thread, releasing device:', self.camID,
              'frame count: %7d\t'%self.count,
              'time  (sec): %7d\t'%int(totalTime),
              '        FPS: %2.2f'%(self.count / totalTime))
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
