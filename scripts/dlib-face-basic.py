import dlib
import numpy as np
import cv2
import time
cap = cv2.VideoCapture(0)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--grey', action='store_true')

args = parser.parse_args()

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

class DeepFaceDetector:
    def __init__(self, path=None):
        import os
        from urllib.request import urlretrieve 
        
        defaultPath = '/tmp/mmod_human_face_detector.dat'
        downloadURL = 'http://users.itk.ppke.hu/~botcs/mmod_human_face_detector.dat'
        
        if path is None:
            print('DNN Face detector path not provided, looking at default path: "%s"' % defaultPath)
            if not os.path.exists(defaultPath):
                print('Downloading DNN Face detector weights from "%s" to "%s"'%(downloadURL, defaultPath))
                urlretrieve(downloadURL, defaultPath)
            
            path = defaultPath
        print('Loading DNN Face detector weights...')
        self.detector = dlib.cnn_face_detection_model_v1(path)
        
        
    def __call__(self, *args, **kwargs):
        mmod_rectangles = self.detector(*args, **kwargs)
        return dlib.rectangles([r.rect for r in mmod_rectangles])


detector = DeepFaceDetector()


i = 0
start_time = time.time()
while(True):
    i += 1
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    if args.grey:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #rects = [r.rect for r in detector(frame, 0)]
    
    rects = detector(frame, 0)
    for rect in rects:
        (x, y, w, h) = rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    FPS = i / (time.time()-start_time)
    print('FPS=%3.3f'%FPS)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
