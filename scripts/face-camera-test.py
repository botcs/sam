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

detector = dlib.get_frontal_face_detector()
i = 0
start_time = time.time()
while(True):
    i += 1
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    if args.grey:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
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
