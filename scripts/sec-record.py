import pdb
import numpy as np
import datetime
import threading
import time
import cv2
import os
import argparse
import sys


parser = argparse.ArgumentParser()
#parser.add_argument('--face', action='store_true')
parser.add_argument('--hd', action='store_true', help='Save in 720p if possible')
parser.add_argument('--length', type=int, default=10, help='Length in seconds to record')
parser.add_argument('--path', default='recordings', help='directory where jpgs will be written to')
parser.add_argument('--cam', nargs='+', type=int, default=[0], help='cam ID that OpenCV can use')
parser.add_argument('--debug', action='store_true', help='Verbose logging of stuff')

args = parser.parse_args()



def isPressed(char, input):
    return ord(char) == input


def TimeStamp(full=True):
    ts = time.time()
    if full:
        return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d.%H-%M-%S.%f')
    else:
        return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d.%H-%M')

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
        self.count = 0

    def capture(self):
        self.captureDevice = cv2.VideoCapture(camID)
        if args.hd:
            self.captureDevice.set(3, 1280)
            self.captureDevice.set(4, 480)

        self.recordDir = os.path.join(args.path, 'camdir-%d' % self.camID)
        self.available = isDeviceWorking(self.captureDevice)
        if not self.available:
            print('WARNING: failed to capture device %d... NOT RECORDING'%camID)
            print('write "ls /dev/video*" to list available devices')
            return False
        print('Recording device %d started, target path:%40s'%(self.camID, self.recordDir))
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
            minuteDir = os.path.join(self.recordDir, TimeStamp(full=False))
            mkdirMaybe(minuteDir)
            path = os.path.join(minuteDir, filename)
            if args.debug:
                print('DEBUG  capture succesful:', ret, ' shape:', frame.shape)

            # Sometimes cv2.imwrite stucks for a while, but the loop must go on
            threading.Thread(target=cv2.imwrite, args=(path, frame)).start()
            if args.debug:
                print('DEBUG  Saved to: %40s' % path)
            self.count += 1

        totalTime = time.time() - startTime
        self.captureDevice.release()
        print('Finishing Recording thread, releasing device:', self.camID,
              'frame count: %7d\t'%self.count,
              'time  (sec): %7d\t'%int(totalTime),
              '        FPS: %2.2f'%(self.count / totalTime))


    def stop(self):
        self._isRunning = False


if __name__ == '__main__':
    threads = []
    print('Starting Recording threads...')
    for camID in args.cam:

        camDir = os.path.join(args.path, 'camdir-%02d' % camID)
        thread = Record(camID)
        thread.start()
        threads.append(thread)

    startTime = time.time()
    totalTime = 0
    while threading.activeCount() > 0 and totalTime < args.length:
        try:
            time.sleep(1.)
        except KeyboardInterrupt:
            print('Manually interruped')
            break
        finally:
            totalTime = time.time() - startTime


    print('Stopping Recording threads...')
    for thread in threads:
        thread.stop()
        thread.join()
    if args.debug:
        print('Devices released succesfully')

    counts = [t.count for t in threads if t.available]
    count = sum(counts)
    print('\n\nTotal statistics',
          'frame count: %7d\t'%count,
          'time  (sec): %7d\t'%int(totalTime),
          '  FPS (avg): %2.2f'%(count / totalTime / len(counts)))
    print('Exiting...')
