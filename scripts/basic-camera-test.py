import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cam', type=int, default=0)
parser.add_argument('--hd', action='store_true', help='Save in 720p if possible')
args = parser.parse_args()

cap = cv2.VideoCapture(args.cam)
if args.hd:
    cap.set(3, 1280)
    cap.set(4, 720)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
