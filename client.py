#!/usr/bin/env python3

## Generally useful libraries
# os - for manipulating paths and files like mv, rm, copy 
# time - for basic benchmark purposes
# argparse - pass arguments from the command line to the script becomes extremely useful 
# pathlib - helps finding the containing directory
import os
from time import time, sleep, ctime
import argparse
import pathlib

# base64 - helps encoding the image buffer to binary strings
# json - data is sent through as binary strings, JSON helps serializing dicts
# threading - required for receieving data asynchronously from the server
import json, pickle
import threading

## Computer vision modules
# cv2 - for capturing web-cam and displaying the live stream
# numpy - for general matrix manipulation of cv2 image arrays
import numpy as np
import cv2
'''
## Sam utility modules
# TODO: naming convention refactor
# these bits provide auxiliary code that implements the following:
#
# prepareOpenFace - neural network architecture, description of the information flow graph
# AlignDlib - Preprocess steps before the face-recognition network. E.g. cropping and rotating faces
# db_query - interface to MySQL server
# ITKGatePirate - interface for communication with specific Wiegand card reader hardware
# drawBBox, drawBanner - display decorators
# getCard2Name - connets CardID to userID for display
# initDB - initialize the MySQL Database

import utils
from utils import prepareOpenFace
from utils import AlignDlib
from utils import rect_to_bb
from utils import db_query
from utils import drawBBox, drawBanner
from utils import CardValidationTracer, PredictionTracer
from utils import getCard2Name, initDB
'''
from utils.streamer import StreamerClient
from client_utils.display import drawBBox, drawBanner

# Knowing where the script is running can be really helpful for setting proper defaults
containing_dir = str(pathlib.Path(__file__).resolve().parent)
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, 'weights')

## Define parameters that can be modified externally
# Routes to essential runtime weights
# Authorization: same name appears in the top K match above the T treshold ratio for C consecutive frames
# Displayed application parameters
## Routes
parser = argparse.ArgumentParser()
parser.add_argument('--embedding-weights', type=str, help='Path to embedding network weights',
                    default=os.path.join(modelDir, 'openface.pth'))
## Display statistic used at server (TAKES NO EFFECT ON THE ACTUAL EVALUATION)
parser.add_argument('--consecutive', type=int, default=5, 
    help='TAKES NO EFFECT ON THE ACTUAL EVALUATION')
parser.add_argument('--k', type=int, help='TAKES NO EFFECT ON THE ACTUAL EVALUATION', default=100)
parser.add_argument('--threshold', type=int, help='TAKES NO EFFECT ON THE ACTUAL EVALUATION', default=50)

parser.add_argument('--display', action='store_true', help='Use OpenCV to show predictions on X')
parser.add_argument('--fullscreen', action='store_true', help='Enable Full Screen display. Only available if --display is used')
parser.add_argument('--card-cooldown', type=int, help='Disable card writer for N secs after each attempt to write', default=3)
parser.add_argument('--virtual', action='store_true', help='Disable card reader')
parser.add_argument('--cam', type=int, default=0, help='Specify video stream /dev/video<cam> to use')
parser.add_argument('--server-address', default='localhost:5555', help='Where to send raw image and card reader data, and receive statistics from. Default: "tcp://198.159.190.163:5555"')
parser.add_argument('--keep-every', type=int, default=1, help='Send every Nth image, discard others.')
args = parser.parse_args()
print('Args parsed:', args)
    


def initializeClient():
    global IS_CLIENT_RUNNING
    global start_time
    global cap
    global it
    global idle_begin
    global pirate
    global streamer
    
    # Initialize webcam before loading every other module
    cap = cv2.VideoCapture(args.cam)
    ret, _ = cap.read()
    if not ret:
        raise RuntimeError('Video capture was unsuccessful.')
    
    IS_CLIENT_RUNNING = True
    start_time = time()
    it = 0
    idle_begin = -1
    
    if not args.virtual:
        from utils.gatepirate import ITKGatePirate
        pirate = ITKGatePirate() 
        
    if args.display:
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        if args.fullscreen:
            cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(
                'frame',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    address, port = args.server_address.split(':')
    port = int(port)
    # Discard older argument only specifies the maximum time that the
    # streamer allows to receieve the message (i.e. deals with network latency)
    # How much time is spent before the data can be read out 
    # from the buffer is not affected
    streamer = StreamerClient(
        (address, port), 
        discard_older=1, 
        only_consecutive=True)
        
    initializeClientVariables()


def initializeClientVariables():
    # These will be updated if server sends new data
    global id_counter
    global BOUNDING_BOXES
    global MAIN_BBOX
    global CARD2NAME
    global OPEN_GATE
    global AUTHORIZED_ID
    global RECOGNIZED_ID
    global TRACED_ID
    global consecutive_occurrence
    
    
    id_counter = None
    BOUNDING_BOXES = []
    MAIN_BBOX = None
    CARD2NAME = {}
    OPEN_GATE = False
    AUTHORIZED_ID = None
    RECOGNIZED_ID = None
    TRACED_ID = None
    consecutive_occurrence = 0
    
    
    
def send(bgrImg, AUTHORIZED_ID):
    #imgString = cv2.imencode('.jpg', bgrImg)[1].tostring()
    
    encoded, buffer = cv2.imencode('.jpg', bgrImg)
    #jpg_as_text = base64.b64encode(buffer)
    jpg_as_text = buffer.tostring()
    
    client_data = {
        'bgrImg': jpg_as_text,
        'AUTHORIZED_ID': AUTHORIZED_ID,
        'message_ts': time()
    }
    message = pickle.dumps(client_data)
    streamer.send(message)
        
    '''
    print('Sent image %15s and ID [%10s] at time: [%10d]'%
        (str(bgrImg.shape), AUTHORIZED_ID, int(time()*1000)))
    '''

def recv():
    '''
    Returns or updates:
      - id_counter: sorted list of (card_id, #occurrence_in_query)
      - BOUNDING_BOXES:       dlib.rectangles of all detected faces
      - MAIN_BBOX:  dlib.rectangle on the closest face
      - CARD2NAME:  dict that maps card_id to shibboleth_id
      - OPEN_GATE:  Boolean, if true client open the registered gate
      - AUTHORIZED_ID: if tracker can trace ID it will be used
      - RECOGNIZED_ID: final suggestion of the face recog. service
      - consecutive_occurrence: # of times RECOGNIZED_ID being the top1
    '''
    global id_counter
    global BOUNDING_BOXES
    global MAIN_BBOX
    global CARD2NAME
    global OPEN_GATE
    global TRACED_ID
    global RECOGNIZED_ID
    global consecutive_occurrence
    
    message = streamer.recv()
    if message is None:
        return
    
    statistics = json.loads(message.decode())

    id_counter = statistics['id_counter']
    BOUNDING_BOXES = statistics['BOUNDING_BOXES']
    MAIN_BBOX = statistics['MAIN_BBOX']
    CARD2NAME = statistics['CARD2NAME']
    OPEN_GATE = statistics['OPEN_GATE']
    TRACED_ID = statistics['TRACED_ID']
    RECOGNIZED_ID = statistics['RECOGNIZED_ID']
    consecutive_occurrence = statistics['consecutive_occurrence']
    
    '''
    return id_counter, BOUNDING_BOXES, MAIN_BBOX, CARD2NAME, OPEN_GATE,\
        AUTHORIZED_ID, RECOGNIZED_ID, consecutive_occurrence
    '''


def asyncRecvLoop():
    global retries
    while IS_CLIENT_RUNNING and streamer.running:
        try:
            recv()
            retries = streamer.retries
        except KeyboardInterrupt:
            print('\nInterrupted manually')
            break
        
        sleep(0.0001)
	
    print('exiting async recv loop')



if __name__ == '__main__':
    initializeClient()
    recvThread = threading.Thread(name='<recv loop thread>', target=asyncRecvLoop)
    recvThread.start()
    print('Begin capture...', ctime())
    while IS_CLIENT_RUNNING:
        it += 1
        

        try:
            # STEP 1: READ IMAGE
            ret, bgrImg = cap.read()
            bgrImg = cv2.flip(bgrImg, 1)
            if not ret:
                raise RuntimeError('Video capture was unsuccessful.')
                
            # STEP 2: READ CARD                
            if not args.virtual:
                CardData = pirate.readCardID(max_age=1000)

            recv()

            # if AUTHORIZED_ID is None:
            # HERE COMES THE CARD ID
            AUTHORIZED_ID = None
            if args.virtual:
                # USE KEY PRESS AS AUTHORIZATION, ID WILL BE THE CHARACTER PRESSED
                pressedKeyCode = cv2.waitKey(10)
                if pressedKeyCode != -1:
                    AUTHORIZED_ID = chr(pressedKeyCode & 255)
            else:
                if len(CardData) == 4:                    
                    AUTHORIZED_ID = CardData[0]

            if AUTHORIZED_ID is not None:
                NAME_ID = CARD2NAME.get(AUTHORIZED_ID)
                NAME_ID = NAME_ID if NAME_ID is not None else AUTHORIZED_ID
                print('READ:\t', NAME_ID, ctime())
            # TODO: Send the frame and AUTHORIZED_ID to server
            if it % args.keep_every == 0:
                threading.Thread(
                    target=send,
                    args=(bgrImg.copy(), AUTHORIZED_ID)
                ).start()
            
            # TODO: Receieve stats from server
            # - id_counter: sorted list of (card_id, #occurrence_in_query)
            # - BBOX:       dlib.rectangles of all detected faces
            # - MAIN_BBOX:  dlib.rectangle on the closest face
            # - CARD2NAME:  dict that maps card_id to shibboleth_id
            # - OPEN_GATE:  Boolean, if true client open the registered gate
            # - AUTHORIZED_ID: if tracker can trace ID it will be used
            # - RECOGNIZED_ID: final suggestion of the face recog. service
            # - consecutive_occurrence: # of times RECOGNIZED_ID being the top1
            if MAIN_BBOX is None:
                if idle_begin < 0: 
                    idle_begin = time()
                idle_time = time() - idle_begin
                FPS = it / (time()-start_time)
                #print('\t\t\tZzzzzz... No face detected (%4.0f sec), FPS:%2.2f\r' %\
                #    (idle_time, FPS), flush=True, end='')
                                    
                if args.display:
                    '''
                    if args.region is not None:
                        # Draw region
                        topleft = (aligner.regionXmin, aligner.regionYmin)
                        bottomright = (aligner.regionXmax, aligner.regionYmax)
                        cv2.rectangle(bgrImg, topleft, bottomright, (255, 255, 255), 3)
                    ''' 
                    bgrImg = drawBanner(bgrImg, retries=retries)
                    cv2.imshow('frame', bgrImg)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

                continue                
            idle_begin = -1

            try:
                if OPEN_GATE:
                    if not args.virtual:
                        pirate.emulateCardID(RECOGNIZED_ID)
                        
                    NAME_ID = CARD2NAME.get(RECOGNIZED_ID)
                    NAME_ID = NAME_ID if NAME_ID is not None else AUTHORIZED_ID
                    print('WRITE:\t', NAME_ID, ctime())
            except KeyError as e:
                print('Catched OPEN error:', e)
                print('LISTING CLIENT VARIABLES...')
                print(id_counter)
                print(BOUNDING_BOXES)
                print(MAIN_BBOX)
                print(CARD2NAME)
                print(OPEN_GATE)
                print(TRACED_ID)
                print(RECOGNIZED_ID)
                print(consecutive_occurrence)
                initializeClientVariables()


            # STEP 7: IF X IS AVAILABLE THEN SHOW FACE BOXES
            if args.display:
                
                # Draw the main bounding box
                for BBOX in BOUNDING_BOXES:
                    if retries == 0:
                        if BBOX == MAIN_BBOX:
                            drawBBox(
                                bgrImg, BBOX, args, id_counter, 
                                consecutive_occurrence, CARD2NAME)
                        else:
                            drawBBox(bgrImg, BBOX, args)
                        
                bgrImg = drawBanner(bgrImg, id_counter, CARD2NAME, TRACED_ID, retries=retries)
                cv2.imshow('frame', bgrImg)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            cap.release()
            print('\nInterrupted manually')
            break

    IS_CLIENT_RUNNING = False
