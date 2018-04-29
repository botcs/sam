import numpy as np
import cv2
from utils import AlignDlib

cap = cv2.VideoCapture(0)
aligner = AlignDlib('weights/shape_predictor_68_face_landmarks.dat')

def rect_to_bb(rect, resize_factor=1):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
    
	# return a tuple of (x, y, w, h)
	return tuple(map(lambda v: resize_factor * v, [x,y,w,h]))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    rgbImg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bbs = aligner.getAllFaceBoundingBoxes(
        img=rgbImg, 
        resize_factor=1,
        forceGrayScale=True)
            
    for rect in bbs:
        (x, y, w, h) = rect_to_bb(rect, 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    aligned_imgs = aligner.multiAlign(rgbImg, imgDim=96, bbs=bbs)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if aligned_imgs is not None: 
        bgrAligned = cv2.cvtColor(np.concatenate(aligned_imgs, axis=0), cv2.COLOR_BGR2RGB)
        cv2.imshow('aligned', bgrAligned)
        print(aligned_imgs.shape)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
