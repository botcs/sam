import dlib
import numpy as np
import cv2
import time 

cap = cv2.VideoCapture(0)

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

detector = cv2.CascadeClassifier('weights/lbpcascade_frontalface.xml')
detector = detector.detectMultiScale
i = 0
start_time = time.time()
while(True):
    i += 1
    # Capture frame-by-frame
    ret, frame = cap.read()


    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1.2)
    end_time = time.time()
    for rect in rects:
        #(x, y, w, h) = rect_to_bb(rect)
        x, y, w, h = rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    FPS = i / (time.time() - start_time)
    print('FPS %3.3f' % FPS) 
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
