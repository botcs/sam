import datetime
import threading
import time
import os
import cv2

def TimeStamp(mode='msec'):
    ts = time.time()
    if mode == 'msec-raw':
        return str(int(time.time()*1000))
    if mode == 'msec':
        return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d.%H-%M-%S.%f')
    if mode == 'minute':
        return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d.%H-%M')
    if mode == 'hour':
        return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d.%H')
    if mode == 'day':
        return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')


def mkdirMaybe(recordDir):
    if not os.path.exists(recordDir):
        os.makedirs(recordDir)
        print('New directory: %40s' % recordDir)



class AsyncSaver(threading.Thread):
    def __init__(self, camID, rootPath):
        self.camID = camID
        self.recordDir = os.path.join(rootPath, 'camdir-%d' % camID)
        
        mkdirMaybe(self.recordDir)
    
    def save(self, bgrImg, bb=None):
        currTS = TimeStamp(mode='msec-raw')
        if bb is None:
            filename = currTS + '.jpg'
        else:
            filename = '%s-%d-%d-%d-%d.jpg' % (currTS, bb.left(), bb.top(), bb.right(), bb.bottom())
        dayDir = os.path.join(self.recordDir, TimeStamp(mode='day'))
        hourDir = os.path.join(dayDir, TimeStamp(mode='hour'))
        minuteDir = os.path.join(hourDir, TimeStamp(mode='minute'))
        mkdirMaybe(minuteDir)
        path = os.path.join(minuteDir, filename)
        threading.Thread(target=cv2.imwrite, args=(path, bgrImg)).start()

