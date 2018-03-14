import numpy as np
import datetime
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

while(True):
    # Capture frame-by-frame
    _, frame0 = cap0.read()
    _, frame1 = cap1.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', np.concatenate([frame0, frame1], axis=1))
    #print(frame0.shape)

    currTS = TimeStamp()
    filename = str(currTS) + ".jpg"
    k = cv2.waitKey(1)
    if isPressed('q', k):
        break

    elif isPressed('h', k):
        path = os.path.join('./h/', filename)
        print('Saved to:%40s'%path)
        cv2.imwrite(path, frame0)

    elif isPressed('j', k):
        path = os.path.join('./j/', filename)
        print('Saved to:%40s'%path)
        cv2.imwrite(path, frame0)

    elif isPressed('k', k):
        path = os.path.join('./k/', filename)
        print('Saved to:%40s'%path)
        cv2.imwrite(path, frame1)

    elif isPressed('l', k):
        path = os.path.join('./l/', filename)
        print('Saved to:%40s'%path)
        cv2.imwrite(path, frame1)

# When everything done, release the capture
cap0.release()
cap1.release()
cv2.destroyAllWindows()
