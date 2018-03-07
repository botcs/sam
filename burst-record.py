import numpy as np
import datetime
import threading
import time
import cv2
import os

cap0 = cv2.VideoCapture(1)
cap1 = cv2.VideoCapture(2)

for cap in cap0, cap1:
    pass
    #cap.set(3, 1280)
    #cap.set(4, 720)

def isPressed(char, input):
    return ord(char) == input

def TimeStamp():
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d.%H-%M-%S.') + str(ts)

def burstRecord(pathToDir, captureDevice, burstLength=.3):
    start = time.time()
    if not os.path.exists(pathToDir):
        os.makedirs(pathToDir)
        print('New directory: %40s' % pathToDir)

    print('Burst started')
    i = 0
    while time.time() - start < burstLength:
        i += 1
        _, frame = captureDevice.read()
        currTS = TimeStamp()
        filename = str(currTS) + '.jpg'
        path = os.path.join(pathToDir, filename)
        print('  Saved to: %40s' % path)
        cv2.imwrite(path, frame)
    print('Recorded %5d frames' % i)



while(True):
    # Capture frame-by-frame
    #_, frame0 = cap0.read()
    #_, frame1 = cap1.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    #cv2.imshow('frame', np.concatenate([frame0, frame1], axis=1))
    cv2.imshow('frame', np.eye(100))
    #print(frame0.shape)

    k = cv2.waitKey(1)
    if isPressed('q', k):
        break

    elif isPressed('h', k):
        threading.Thread(target=burstRecord, args=('h', cap0)).start()

    elif isPressed('j', k):
        threading.Thread(target=burstRecord, args=('j', cap1)).start()


# When everything done, release the capture
cap0.release()
cap1.release()
