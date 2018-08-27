#!/usr/bin/env python3

## Generally useful libraries
# os - for manipulating paths and files like mv, rm, copy 
# time - for basic benchmark purposes
# argparse - pass arguments from the command line to the script becomes extremely useful 
# pathlib - helps finding the containing directory
import os
from time import time
import argparse
import pathlib

# base64 - helps encoding the image buffer to binary strings
# json - data is sent through as binary strings, JSON helps serializing dicts
# threading - required for receieving data asynchronously from the server
# zmq - communication with the server
import base64
import json, pickle
import threading
import zmq
# Required if socket is used in a thread
import zmq.eventloop.ioloop
zmq.eventloop.ioloop.install()

## Computer vision modules
# torch - for neural network and GPU accelerated processes
# cv2 - for capturing web-cam and displaying the live stream
# numpy - for general matrix manipulation of cv2 image arrays
import numpy as np
import dlib
import torch
import cv2

## pytorch utility functions
# FloatTensor - is set as the default Tensor type when recasting, easy to switch to half-precision
# ToTensor - takes a numpy array and converts it to a torch array while normalizing it as well
from torch.cuda import FloatTensor as Tensor
from torchvision.transforms import ToTensor

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
from utils import CardValidationTracer, PredictionTracer
from utils import getCard2Name, initDB


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
parser.add_argument('--database', type=str, help='path to embedding->name database',
                    default=os.path.join(modelDir, 'REALTIME-DB.tar'))
parser.add_argument('--dlib-face-predictor', type=str, help='Path to dlib\'s face predictor.',
                    default=os.path.join(modelDir, 'shape_predictor_68_face_landmarks.dat'))


## Auth
parser.add_argument('--consecutive', type=int, default=30, 
    help='How many frames is required to be authorized as the same person')
parser.add_argument('--k', type=int, help='List top K results', default=100)
parser.add_argument('--threshold', type=int, help='Threshold for opening count in %%', default=50)

## Display
parser.add_argument('--region', type=int, nargs=4, help='detect face only in [Xmin Ymin Width Height] region')
parser.add_argument('--display', action='store_true', help='Use OpenCV to show predictions on X')
parser.add_argument('--fullscreen', action='store_true', help='Enable Full Screen display. Only available if --display is used')
parser.add_argument('--card-cooldown', type=int, help='Disable card writer for N secs after each attempt to write', default=3)
parser.add_argument('--virtual', action='store_true', help='Disable card reader')
parser.add_argument('--cam', type=int, default=0, help='Specify video stream /dev/video<cam> to use')
parser.add_argument('--port', type=int, default=5555, help='Where to send raw image and card reader data, and receive statistics from. Default: 5555')
#parser.add_argument('--process-every', type=int, default=1, help='process every Nth frame and discard others')
parser.add_argument('--discard-older', type=int, default=500, help='discard frames older than N msec')
parser.add_argument('--sql-buffer-size', type=int, default=50, help='Uploading buffered images to the database may take some time. Large size will occur slowdon less frequently but for more time. Small buffer size will trigger SQL sync more often, but the process will be shorter. Opt with regards to the actual bandwith.')
parser.add_argument('--sql', action='store_true', help='if NOT set then no attempts will be made to sync with the DB')
args = parser.parse_args()

#TODO: pretty print arguments
#for k, v in args.items():
#    print(k,':',v)
   
print('arsg:', args)
# PyTorch version check
v3 = torch.__version__ == '0.3.1'
    
def initializeServer():
    global IS_SERVER_RUNNING
    global start_time
    global it
    global pirate
    global server_socket
    
    # These will be sent to client
    global id_counter
    global DLIB_BOUNDING_BOXES
    global DLIB_MAIN_BBOX
    global CARD2NAME
    global OPEN_GATE
    global AUTHORIZED_ID
    global RECOGNIZED_ID
    global consecutive_occurrence
    
    # Face recognition service variables
    global KNOWN_DB
    global net
    global aligner
    global cardTracer
    global predTracer
    global use_cuda
    global tensor_converter
    global pdist
    global last_cardwrite
        
    IS_SERVER_RUNNING = True
    start_time = time()
    it = 0
    
    context = zmq.Context()
    server_socket = context.socket(zmq.PAIR)
    server_socket.bind('tcp://*:%d'%args.port)
    
    id_counter = None
    DLIB_BOUNDING_BOXES = None
    DLIB_MAIN_BBOX = None
    CARD2NAME = {}
    OPEN_GATE = False
    AUTHORIZED_ID = None
    RECOGNIZED_ID = None
    consecutive_occurrence = 0
    
    if args.sql:    
        initDB()

    
    KNOWN_DB = {'emb':Tensor(0, 128), 'id':[]}
    if args.sql:
        CARD2NAME = getCard2Name()
    if args.database is not None:
        KNOWN_DB = torch.load(args.database)
        
        # Torch 0.3.1 legacy stuff
        if v3:
            if isinstance(KNOWN_DB['emb'], torch.autograd.Variable): # LEGACY LINE
                KNOWN_DB['emb'] = KNOWN_DB['emb'].data # LEGACY LINE
        
            
    print('Size of database: %5d samples' % len(KNOWN_DB['emb']))     
    net = prepareOpenFace()
    net = net.eval()
    net.load_state_dict(torch.load(args.embedding_weights))
    print('Model loaded')

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        print('CUDA is available, uploading parameters to device...')
        
        net.cuda()
        print('Neural Network OK')

        KNOWN_DB['emb'] = KNOWN_DB['emb'].cuda()
        print('Embedding database OK')
    else:
        print('CUDA is not available')
    
    
    # aligner takes a fullframe and returns a cropped > aligned (warped) image
    aligner = AlignDlib(facePredictor=args.dlib_face_predictor, region=args.region)
    
    
    # tracer handles online training and ID assignment
    cardTracer = CardValidationTracer(SQLBufferSize=args.sql_buffer_size)
    predTracer = PredictionTracer(SQLBufferSize=args.sql_buffer_size)
    
    
    # tensor converter takes a numpy array and returns a normalized Torch Tensor 
    tensor_converter = ToTensor()
    
    # pdist defines the metric which will be used for queries
    pdist = torch.nn.PairwiseDistance(p=2)
    
    # Cooldown counter for emitting the OPEN_GATE signal
    last_cardwrite = time()
    
    
    
    
def send():
    # Eliminate dlib dependency on client
    # and reduce message size
    BOUNDING_BOXES = [rect_to_bb(rect) for rect in DLIB_BOUNDING_BOXES]
    if DLIB_MAIN_BBOX is not None:
        MAIN_BBOX = rect_to_bb(DLIB_MAIN_BBOX)
    else:
        MAIN_BBOX = None
    sendData = {
        'id_counter': id_counter,
        'BOUNDING_BOXES': BOUNDING_BOXES,
        'MAIN_BBOX': MAIN_BBOX,
        'CARD2NAME': CARD2NAME,
        'OPEN_GATE': OPEN_GATE,
        'AUTHORIZED_ID': AUTHORIZED_ID,
        'RECOGNIZED_ID': RECOGNIZED_ID,
        'consecutive_occurrence': consecutive_occurrence,
        'message_ts': time()
    }
    message = json.dumps(sendData)
    
    lock = threading.RLock()
    #lock.acquire()
    try:
        server_socket.send_string(message)
    except RuntimeError as e:
        print('SERVER <send> ERROR: ', e)
        
    finally:
        pass
        #lock.release()



def recv():
    lock = threading.RLock()
    #lock.acquire()
    try:
        message = server_socket.recv()
        client_data = pickle.loads(message)
        message_ts = client_data['message_ts']
        AUTHORIZED_ID = client_data['AUTHORIZED_ID']
        
        jpg_as_text = client_data['bgrImg']
        img = base64.b64decode(jpg_as_text)
        img = jpg_as_text
        img = np.fromstring(img, dtype=np.uint8)
        bgrImg = cv2.imdecode(img, cv2.IMREAD_COLOR)
        
        
    except RuntimeError as e:
        print('SERVER <recv> ERROR: ', e)
    finally:
        pass
        #lock.release()
    delay_time = (time() - message_ts)*1000 
    FPS = it / (time() - start_time)
    print('Receieved image #%010d and ID [%10s] with delay [%4.0f] msec, Avg. FPS = %3.1f'%
        (it, AUTHORIZED_ID, delay_time, fps_counter.ema_fps))
    return bgrImg, AUTHORIZED_ID, delay_time


class FPSCounter():
    def __init__(self):
        self.start_time = time()
        self.last_call = time()
        self.prev_call = time()
        self.frame_count = 0

        self.ema_fps = None

    def __call__(self):
        self.frame_count += 1
        self.prev_call = self.last_call
        self.last_call = time()
        self.update_ema()

    def update_ema(self, alpha=0.1):
        current_fps = 1 / (self.last_call - self.prev_call)

        if self.ema_fps is None:
            self.ema_fps = current_fps
        
        self.ema_fps = alpha * current_fps + (1-alpha) * self.ema_fps

        


if __name__ == '__main__':

    initializeServer()
    print('Starting service...')
    if not v3: 
        torch.no_grad().__enter__()

    fps_counter = FPSCounter()
    while IS_SERVER_RUNNING:
        
        # Only flush when the server is idle
        #cardTracer.flush()
        #predTracer.flush()
        fps_counter()
        try:
            # STEP 1: READ IMAGE
            # STEP 2: READ CARD                
            bgrImg, AUTHORIZED_ID, delay_time = recv()
            if delay_time > args.discard_older:
                continue
            '''
            if it % args.process_every != 0:
                continue
            '''            
             
            it += 1
            #FPS = it / (time()-start_time)
            DLIB_BOUNDING_BOXES = aligner.getAllFaceBoundingBoxes(bgrImg)
            DLIB_MAIN_BBOX = aligner.extractLargestBoundingBox(DLIB_BOUNDING_BOXES)
            
            if DLIB_MAIN_BBOX is None:
                if args.sql:
                    cardTracer.flush()
                    predTracer.flush()

                threading.Thread(target=send).start()
                continue

            
            
            # STEP 2: PREPROCESS IMAGE
            rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
            img = rgbImg
            aligned_img = aligner.align(96, img, bb=DLIB_MAIN_BBOX)            
            
            x = tensor_converter(aligned_img)
            x.requires_grad = False
            x = x[None]
            if use_cuda:
                x = x.cuda()
                if v3:
                    x = torch.autograd.Variable(x, volatile=True, requires_grad=False) # LEGACY LINE
            
            # STEP 3: EMBEDD IMAGE
            inference_start = time()
            embedding128 = net(x)[0]
            if v3:
                embedding128 = embedding128.data # LEGACY LINE
            inference_time = time() - inference_start            

            # STEP 4: COMPARE TO REGISTERED EMBEDDINGS
            if len(KNOWN_DB['emb']) > 0:
                topk_start = time()
                distances = pdist(KNOWN_DB['emb'], embedding128.expand_as(KNOWN_DB['emb']))
                distances.squeeze_()
                sorted_distances, idxs = torch.sort(distances)
                sorted_distances = sorted_distances[:args.k]
                idxs = idxs[:args.k]
                topk_time = time() - topk_start
                
                count_start = time()
                id_counter = {}
                for idx in idxs:
                    n = KNOWN_DB['id'][idx]
                    if id_counter.get(n) is None:
                        id_counter[n] = 1
                    else:
                        id_counter[n] += 1
                id_counter = sorted(
                    list(id_counter.items()), 
                    key=lambda x: x[1], reverse=True)[:args.k]
                    
                count_time = time() - count_start
            else:
                id_counter = [('<UNK>', 100)]

            # STEP 6: TRACKING:
            AUTHORIZED_ID, KNOWN_DB = cardTracer.track(
                bgrImg=bgrImg.copy(), 
                mainBB=DLIB_MAIN_BBOX, 
                embedding128=embedding128, 
                AUTHORIZED_ID=AUTHORIZED_ID, 
                KNOWN_DB=KNOWN_DB, 
                virtual=args.virtual)

            #print('\r\tEmbedding network inference time: %1.4f sec, FPS=%2.2f' % (inference_time, FPS), end='')
            # STEP 5: POLICY FOR OPENING THE TURNSPIKE
            # RECOGNIZED_ID has to be present for a certain amount of time
            # until it is validated by the policy, if RECOGNIZED_ID changes
            # even once, it will discard previous record
            if (id_counter[0][0] != '<UNK>' and 
                id_counter[0][1]/args.k *100 > args.threshold and 
                RECOGNIZED_ID == id_counter[0][0]):
                # Prediction from a previous frame has been validated because:
                # the card_id is not <UNK>
                # in the last embedded query the #occurence of the closest ID is above the treshold
                # and the previously RECOGNIZED id is the same as the current candidate
                consecutive_occurrence += 1
            else:
                # Previous RECOGNIZED_ID had been discarded
                # reset the consecutive counter to 0
                # assign the new ID
                RECOGNIZED_ID = id_counter[0][0]
                consecutive_occurrence = 0
                
            # The candidate person is RECOGNIZED
            if consecutive_occurrence >= args.consecutive:
                readyToEmulate = (time() - last_cardwrite) > args.card_cooldown
                name_id = CARD2NAME.get(RECOGNIZED_ID)
                
                OPEN_GATE = False
                if name_id is not None:
                    predTracer.addPrediction(bgrImg.copy(), DLIB_MAIN_BBOX, RECOGNIZED_ID)
                    if readyToEmulate:
                        print('OPEN:', name_id, RECOGNIZED_ID, time())
                        #pirate.emulateCardID(RECOGNIZED_ID)
                        OPEN_GATE = True
                        last_cardwrite = time()
                    
                elif readyToEmulate:
                    print('Would open, but ID is not registered', RECOGNIZED_ID)
                    last_cardwrite = time()
            
            
            
            # STEP 8:
            # TODO: Async update of CARD2NAME
            if args.sql and it % 50 == 0:
                CARD2NAME = getCard2Name()
            
            
            # STEP N:
            threading.Thread(target=send).start()

        except KeyboardInterrupt:
            print('\nInterrupted manually')
            break
        
    IS_SERVER_RUNNING = False
    # FINALLY: Save the learned representations
    # torch.save(KNOWN_DB, os.path.join(modelDir, 'REALTIME-DB.tar'))
    
        
            
