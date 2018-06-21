#!/usr/bin/env python3

## Generally useful libraries
# os - for manipulating paths and files like mv, rm, copy 
# time - for basic benchmark purposes
# argparse - pass arguments from the command line to the script becomes extremely useful 
# pathlib - helps finding the containing directory
import os
import time
import argparse
import pathlib

## Computer vision modules
# torch - for neural network and GPU accelerated processes
# cv2 - for capturing web-cam and displaying the live stream
# numpy - for general matrix manipulation of cv2 image arrays
import numpy as np
import dlib
#DLIB_CNN = dlib.cnn_face_detection_model_v1('/tmp/mmod_human_face_detector.dat')
#print(DLIB_CNN(np.zeros([480, 640, 3], dtype='uint8')))
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
from utils import ITKGatePirate
from utils import drawBBox, drawBanner
from utils import Tracer
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
                    default=os.path.join(modelDir, 'NEGATIVE_DATABASE.tar'))
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
args = parser.parse_args()
print('Args parsed:', args)
    

if __name__ == '__main__':
    
    # Initialize webcam before loading every other module
    cap = cv2.VideoCapture(0)
    ret, _ = cap.read()
    if not ret:
        raise RuntimeError('Video capture was unsuccessful.')

    
    initDB('/home/botoscs/sam/utils/db.conf')

    
    KNOWN_DB = {'emb':Tensor(0, 128), 'id':[]}
    CARD2NAME = getCard2Name()
    if args.database is not None:
        KNOWN_DB = torch.load(args.database)
        
    AUTHORIZED_ID = None
    
    if args.display:
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        if args.fullscreen:
            cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(
                'frame',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    if not args.virtual:
        pirate = ITKGatePirate()    
        
    
    print('Size of database: %5d samples' % len(KNOWN_DB['emb']))     

    net = prepareOpenFace()
    net = net.eval()
    net.load_state_dict(torch.load(args.embedding_weights))
    pdist = torch.nn.PairwiseDistance(p=2)
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
    aligner = AlignDlib(facePredictor=args.dlib_face_predictor, region=None)
    
    
    # tracer handles online training and ID assignment
    tracer = Tracer(x_displacement_treshold=100, SQLBufferSize=5)
    
    # tensor converter takes a numpy array and returns a normalized Torch Tensor 
    tensor_converter = ToTensor()
    
    # Initializing the face recognition application parameters
    last_cardwrite = time.time()
    it = 0
    start_time = time.time()
    idle_begin = -1
    RECOGNIZED_ID = None
    consecutive_occurrence = 0
    print('Begin capture')
    torch.no_grad().__enter__()
    while True:
        it += 1
        tracer.flush()           

        try:
            # STEP 1: READ IMAGE
            ret, bgrImg = cap.read()
            bgrImg = cv2.flip(bgrImg, 1)
            if not ret:
                raise RuntimeError('Video capture was unsuccessful.')
                
            BOUNDING_BOXES = aligner.getAllFaceBoundingBoxes(bgrImg)
            MAIN_BBOX = aligner.extractLargestBoundingBox(BOUNDING_BOXES)
            
            if MAIN_BBOX is None:
                if idle_begin < 0: 
                    idle_begin = time.time()
                idle_time = time.time() - idle_begin
                FPS = it / (time.time()-start_time)
                #print('\t\t\tZzzzzz... No face detected (%4.0f sec), FPS:%2.2f\r' %\
                #    (idle_time, FPS), flush=True, end='')
                
                if args.display:
                    if args.region is not None:
                        # Draw region
                        topleft = (aligner.regionXmin, aligner.regionYmin)
                        bottomright = (aligner.regionXmax, aligner.regionYmax)
                        cv2.rectangle(bgrImg, topleft, bottomright, (255, 255, 255), 3)
                        
                    bgrImg = drawBanner(bgrImg)
                    cv2.imshow('frame', bgrImg)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break     
                        
                continue
                
            idle_begin = -1
          

            # STEP 2: PREPROCESS IMAGE
            rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
            img = rgbImg
            aligned_img = aligner.align(96, img, bb=MAIN_BBOX)            
            
            x = tensor_converter(aligned_img)
            x.requires_grad = False
            x = x[None]
            if use_cuda:
                x = x.cuda()
            
            # STEP 3: EMBEDD IMAGE
            inference_start = time.time()
            embedding128 = net(x)[0]
            inference_time = time.time() - inference_start            

           
            # STEP 4: COMPARE TO REGISTERED EMBEDDINGS
            if len(KNOWN_DB['emb']) > 0:
                topk_start = time.time()
                distances = pdist(KNOWN_DB['emb'], embedding128.expand_as(KNOWN_DB['emb']))
                sorted_distances, idxs = torch.sort(distances)
                sorted_distances = sorted_distances[:args.k]
                idxs = idxs[:args.k]
                topk_time = time.time() - topk_start
                
                count_start = time.time()
                id_counter = {}
                for idx in idxs.data:
                    n = KNOWN_DB['id'][idx]
                    if id_counter.get(n) is None:
                        id_counter[n] = 1
                    else:
                        id_counter[n] += 1
                id_counter = sorted(list(id_counter.items()), key=lambda x: x[1], reverse=True)[:args.k]
                count_time = time.time() - count_start
            else:
                id_counter = [('<UNK>', 100)]
          
 
            # STEP 5: OPEN TURNSPIKE
            # TODO: design a good policy
            if (id_counter[0][0] != '<UNK>' and 
                id_counter[0][1]/args.k *100 > args.threshold and 
                RECOGNIZED_ID == id_counter[0][0]):
                
                consecutive_occurrence += 1
                
                if (not args.virtual and 
                    consecutive_occurrence >= args.consecutive and 
                    (time.time() - last_cardwrite) > args.card_cooldown):
                    pirate.emulateCardID(id_counter[0][0])
                    last_cardwrite = time.time()
                    '''
                    card_id = getSQLcardID(RECOGNIZED_ID)
                    if card_id is not None:
                        print('OPEN:', RECOGNIZED_ID, card_id)
                        SQLInsert(card_id)
                        if not args.virtual:
                            pirate.emulateCardID(card_id)
                    '''    
                        

            else:
                RECOGNIZED_ID = id_counter[0][0]
                consecutive_occurrence = 0
            
                
            # STEP 6: (RETARDED-)SMART TRACKING:
            AUTHORIZED_ID, KNOWN_DB = tracer.track(
                bgrImg=bgrImg.copy(), 
                mainBB=MAIN_BBOX, 
                embedding128=embedding128, 
                AUTHORIZED_ID=AUTHORIZED_ID, 
                KNOWN_DB=KNOWN_DB, 
                virtual=args.virtual)

            if not args.virtual:        
                CardData = pirate.readCardID(max_age=1000)
            if AUTHORIZED_ID is None:
                # HERE COMES THE CARD ID
                if args.virtual:
                    # USE KEY PRESS AS AUTHORIZATION, ID WILL BE THE CHARACTER PRESSED
                    pressedKeyCode = cv2.waitKey(10)
                    if pressedKeyCode != -1:
                        AUTHORIZED_ID = chr(pressedKeyCode & 255)
                else:
                    if len(CardData) == 4:
                        AUTHORIZED_ID = CardData[0]
            
            FPS = it / (time.time()-start_time)
            #print('\r\tEmbedding network inference time: %1.4f sec, FPS=%2.2f' % (inference_time, FPS), end='')
            
            # STEP 7: IF X IS AVAILABLE THEN SHOW FACE BOXES
            if args.display:
                
                # Draw the main bounding box
                for BBOX in BOUNDING_BOXES:
                    if BBOX == MAIN_BBOX:
                        drawBBox(bgrImg, BBOX, args, id_counter, consecutive_occurrence, CARD2NAME)
                    else:
                        drawBBox(bgrImg, BBOX, args)
                
                bgrImg = drawBanner(bgrImg, id_counter, CARD2NAME, AUTHORIZED_ID)
                cv2.imshow('frame', bgrImg)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
              
            # STEP 8:
            # TODO: Async update of CARD2NAME
            if it % 50 == 0:
                CARD2NAME = getCard2Name()
 

        except KeyboardInterrupt:
            print('Interrupted manually')
            break
            
    
    # FINALLY: Save the learned representations
    torch.save(KNOWN_DB, os.path.join(modelDir, 'REALTIME-DB.tar'))
        
            
