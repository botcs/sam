import dlib
import numpy as np
import cv2
import time
cap = cv2.VideoCapture(0)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gray', action='store_true')
parser.add_argument('--region', type=int, nargs=4)
parser.add_argument('--display', action='store_true')
args = parser.parse_args()

if args.region:
    xmin = args.region[0]
    ymin = args.region[1]
    xmax = xmin + args.region[2]
    ymax = ymin + args.region[3]



detector = dlib.get_frontal_face_detector()
i = 0
start_time = time.time()
while(True):
    i += 1
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    if args.gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if args.region:        
        peephole = frame[ymin:ymax, xmin:xmax]
        rel_rects = detector(peephole, 0)
        rects = []
        for r in rel_rects:
            rects.append(dlib.rectangle(r.left()+xmin, r.top()+ymin, r.right()+xmin, r.bottom()+ymin))
        # Draw peephole's bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)        
    else:
        rects = detector(frame)


    
    for rect in rects:
        cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)
    
    FPS = i / (time.time()-start_time)
    print('FPS=%3.3f'%FPS)
    if args.display:
        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
