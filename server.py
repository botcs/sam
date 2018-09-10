#!/usr/bin/env python3

## Generally useful libraries
# os - for manipulating paths and files like mv, rm, copy 
# time - for basic benchmark purposes
# argparse - pass arguments from the command line to the script becomes extremely useful 
# pathlib - helps finding the containing directory
import os
from time import time, sleep
import argparse
import pathlib

# base64 - helps encoding the image buffer to binary strings
# json - data is sent through as binary strings, JSON helps serializing dicts
# threading - required for receieving data asynchronously from the server
import json, pickle
import threading

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
from utils.openface import prepareOpenFace
from utils.align_dlib import AlignDlib, rect_to_bb
from utils.tracer import CardValidationTracer, PredictionTracer
from utils.sqlrequest import db_query, getCard2Name, initDB
from utils.streamer import StreamerServer

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
parser.add_argument('--consecutive', type=int, default=5, 
    help='How many frames is required to be authorized as the same person')
parser.add_argument('--k', type=int, help='List top K results', default=100)
parser.add_argument('--threshold', type=int, help='Threshold for opening count in %%', default=50)

## Display
parser.add_argument('--display', '-d', action='store_true', help='Use OpenCV to show predictions on X')
parser.add_argument('--fullscreen', '-x', action='store_true', help='Enable Full Screen display. Only available if --display is used')
parser.add_argument('--card-cooldown', type=int, help='Disable card writer for N secs after each attempt to write', default=3)
parser.add_argument('--region', type=int, nargs=4, help='detect face only in [Xmin Ymin Width Height] region, deprecated as fuck')
parser.add_argument('--virtual', action='store_true', help='Disable saving embedding database')
parser.add_argument('--cam', type=int, default=0, help='Specify video stream /dev/video<cam> to use')
parser.add_argument('--port', type=int, default=5555, help='Where to send raw image and card reader data, and receive statistics from. Default: 5555')
#parser.add_argument('--process-every', type=int, default=1, help='process every Nth frame and discard others')
parser.add_argument('--discard-older', '-D', type=int, default=200, help='discard frames older than N msec')
parser.add_argument('--sql-buffer-size', type=int, default=10, help='Uploading buffered images to the database may take some time. Large size will occur slowdon less frequently but for more time. Small buffer size will trigger SQL sync more often, but the process will be shorter. Opt with regards to the actual bandwith.')
parser.add_argument('--sql', '-Q', action='store_true', help='if NOT set then no attempts will be made to sync with the DB')
parser.add_argument('--verbose', '-v', action='store_true', help='Help benchmarking and debugging')
args = parser.parse_args()

#TODO: pretty print arguments
#for k, v in args.items():
#    print(k,':',v)
   
print('arsg:', args)
# PyTorch version check
v3 = torch.__version__ == '0.3.1'
    
def loadEmbeddingDB():
    global KNOWN_DB
    if args.database is not None:
        KNOWN_DB = torch.load(args.database)
        
        # Torch 0.3.1 legacy stuff
        if v3:
            if isinstance(KNOWN_DB['emb'], torch.autograd.Variable): # LEGACY LINE
                KNOWN_DB['emb'] = KNOWN_DB['emb'].data # LEGACY LINE
        print('Updated embedding database from: %10s'%args.database, '%5d samples' % len(KNOWN_DB['emb']))  
    
    
    
def initializeServer():
    global IS_SERVER_RUNNING
    global start_time
    global it
    global pirate
    global streamer
    
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
    compute_time = 0
    it = 0
    
    '''
    context = zmq.Context()
    server_socket = context.socket(zmq.PAIR)
    server_socket.bind('tcp://*:%d'%args.port)
    '''
    
    
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
    loadEmbeddingDB()
    if args.sql:
        CARD2NAME = getCard2Name()
        
       
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
    
    address = 'localhost'
    port = args.port
    # Discard older argument only specifies the maximum time that the
    # streamer allows to receieve the message (i.e. deals with network latency)
    # How much time is spent before the data can be read out 
    # from the buffer is not affected
    streamer = StreamerServer(
        (address, port), 
        discard_older=args.discard_older, 
        only_consecutive=True)
    
    
def send(sendData):
    # Eliminate dlib dependency on client
    # and reduce message size
    '''
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
    '''
    message = json.dumps(sendData)
    
    try:
        streamer.send(message)
    except RuntimeError as e:
        print('SERVER <send> ERROR: ', e)
        


def recv():
    def recv_msg():
        try:
            message = streamer.recv()
            while message is None:        
                message = streamer.recv()
                sleep(.0001)
            client_data = pickle.loads(message)
            message_ts = client_data['message_ts']
            AUTHORIZED_ID = client_data['AUTHORIZED_ID']
            
            jpg_as_text = client_data['bgrImg']
            #img = base64.b64decode(jpg_as_text)
            img = jpg_as_text
            img = np.fromstring(img, dtype=np.uint8)
            bgrImg = cv2.imdecode(img, cv2.IMREAD_COLOR)

            
        except RuntimeError as e:
            print('SERVER <recv> ERROR: ', e)
        finally:
            pass
        delay_time = int((time() - message_ts)*1000)
        return bgrImg, AUTHORIZED_ID, delay_time

    while IS_SERVER_RUNNING and streamer.running:
        bgrImg, AUTHORIZED_ID, delay_time = recv_msg()
        
        keepImg = delay_time < args.discard_older or AUTHORIZED_ID is not None
        utilization = effective_fps.ema_fps / compute_fps.ema_fps
        status_log = '+' if keepImg else 'o'
        status_log += ' it: %05d' % it
        status_log += ' ID: [%8s]' % AUTHORIZED_ID
        status_log += ' delay: %4dms' % delay_time
        status_log += ' FPS: %5.1f' % effective_fps.ema_fps
        status_log += ' dtct: %5.1f' % face_detector_fps.ema_fps
        status_log += ' reco: %5.1f' % face_recognition_fps.ema_fps
        status_log += ' stat: %5.1f' % face_stat_fps.ema_fps
        status_log += ' comp: %5.1f' % compute_fps.ema_fps
        status_log += ' UTIL: %2.1f%%' % (utilization*100)
        
        if args.verbose:
            print(status_log)

        if keepImg:
            break
    
    return bgrImg, AUTHORIZED_ID, delay_time


class FPSCounter():
    def __init__(self):
        self.start_time = time()
        self.last_call = time()
        self.prev_call = time()
        self.count = 0

        self.ema_fps = -1.

    def __call__(self):
        self.tak()

    def update_ema(self, alpha=0.05):
        current_fps = 1 / (self.last_call - self.prev_call)

        if self.count == 1:
            self.ema_fps = current_fps
        
        self.ema_fps = alpha * current_fps + (1-alpha) * self.ema_fps

    def tik(self):
        self.last_call = time()
        
    def tak(self):
        self.count += 1
        self.prev_call = self.last_call
        self.last_call = time()
        self.update_ema()
        return self.ema_fps
        

        

if __name__ == '__main__':

    initializeServer()
    print('Starting service...')
    if not v3: 
        torch.no_grad().__enter__()

    effective_fps = FPSCounter()
    face_detector_fps = FPSCounter()
    face_recognition_fps = FPSCounter()
    face_stat_fps = FPSCounter()
    compute_fps = FPSCounter()
    while IS_SERVER_RUNNING:

        try:
            
            
        
            # STEP 1: READ IMAGE
            # STEP 2: READ CARD                
            bgrImg, AUTHORIZED_ID, delay_time = recv()
            '''
            if it % args.process_every != 0:
                continue
            '''            
             
            it += 1
            effective_fps()
            compute_fps.tik() # THROUGHPUT OF COMPUTER BEGIN (Regardless input)
            # STEP 8:
            # TODO: Async update of CARD2NAME
            if args.sql and it % 70 == 0:
                CARD2NAME = getCard2Name()
                
            
            
            
            face_detector_fps.tik() # FACE DETECTION TIMER BEGIN
            DLIB_BOUNDING_BOXES = aligner.getAllFaceBoundingBoxes(bgrImg)
            DLIB_MAIN_BBOX = aligner.extractLargestBoundingBox(DLIB_BOUNDING_BOXES)
            face_detector_fps.tak() # FACE DETECTION TIMER END
            
            # STEP N:
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
                'TRACED_ID': cardTracer.traced_id,
                'RECOGNIZED_ID': RECOGNIZED_ID,
                'consecutive_occurrence': consecutive_occurrence,
                'message_ts': time()
            }
            threading.Thread(target=send, args=[sendData]).start()
            
            if DLIB_MAIN_BBOX is None:
                if it % 1000 == 0:
                    loadEmbeddingDB()
                
                if args.sql:
                    cardTracer.flush()
                    predTracer.flush()
                
                compute_fps.tak() # COMPUTE TIME END
                continue
                

            
            
            # STEP 2: PREPROCESS IMAGE
            face_recognition_fps.tik() # FACE RECOGNITION TIMER BEGIN
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
            #face_recognition_fps.tik() # FACE RECOGNITION TIMER BEGIN
            embedding128 = net(x)[0]
            if v3:
                embedding128 = embedding128.data # LEGACY LINE    
            face_recognition_fps.tak() # FACE RECOGNITION TIMER END
            
            # STEP 4: COMPARE TO REGISTERED EMBEDDINGS
            face_stat_fps.tik() # FACE STATISTICS TIMER BEGIN
            if len(KNOWN_DB['emb']) > 0:
                distances = pdist(KNOWN_DB['emb'], embedding128.expand_as(KNOWN_DB['emb']))
                distances.squeeze_()
                sorted_distances, idxs = torch.sort(distances)
                sorted_distances = sorted_distances[:args.k]
                idxs = idxs[:args.k]
                
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
                    
            else:
                id_counter = [('<UNK>', 100)]

            # STEP 6: TRACKING:
            
            KNOWN_DB = cardTracer.track(
                bgrImg=bgrImg.copy(), 
                mainBB=DLIB_MAIN_BBOX, 
                embedding128=embedding128, 
                AUTHORIZED_ID=AUTHORIZED_ID, 
                KNOWN_DB=KNOWN_DB, 
                virtual=args.virtual)
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
            OPEN_GATE = False
            if consecutive_occurrence >= args.consecutive:
                readyToEmulate = (time() - last_cardwrite) > args.card_cooldown
                name_id = CARD2NAME.get(RECOGNIZED_ID)
                
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
                    
            face_stat_fps.tak() # FACE STATISTICS TIMER END
            compute_fps.tak() # COMPUTE TIME END
            

        except KeyboardInterrupt:
            print('\nInterrupted manually')
            break
        
    IS_SERVER_RUNNING = False
    # FINALLY: Save the learned representations
    if not args.virtual:
        torch.save(KNOWN_DB, args.database)
    
        
            
