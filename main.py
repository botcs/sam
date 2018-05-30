#!/usr/bin/env python3

import utils
import os
import time
import argparse
import pathlib
import torch
import cv2
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from utils import prepareOpenFace, AlignDlib, rect_to_bb, send_query, ITKGatePirate, AsyncSaver


containing_dir = str(pathlib.Path(__file__).resolve().parent)

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, 'weights')

parser = argparse.ArgumentParser()

parser.add_argument('--dlib', type=str, help='Path to dlib\'s face predictor.',
                    default=os.path.join(modelDir, 'shape_predictor_68_face_landmarks.dat'))
parser.add_argument('--embedding-weights', type=str, help='Path to embedding network weights',
                    default=os.path.join(modelDir, 'openface.pth'))
parser.add_argument('--database', type=str, help='path to embedding2name database',
                    default=os.path.join(modelDir, 'DEPLOY_DATABASE.tar'))
parser.add_argument('--k', type=int, help='List top K results', default=100)
parser.add_argument('--threshold', type=int, help='Threshold for opening count in %%', default=50)
parser.add_argument('--consecutive', type=int, help='How many frames is required to be authorized as the same person', 
                    default=30)
                    
parser.add_argument('--gray', action='store_true')
parser.add_argument('--region', type=int, nargs=4, help='detect face only in [Xmin Ymin Width Height] region', default=[200, 100, 150, 150])
parser.add_argument('--display', action='store_true', help='Use OpenCV to show predictions on X')
parser.add_argument('--virtual', action='store_true', help='Disable card reader')
parser.add_argument('--fullscreen', action='store_true', help='Enable Full Screen display. Only available if --display is used')
parser.add_argument('--card-cooldown', type=int, help='Disable card writer for N secs after each attempt to write', default=3)

args = parser.parse_args()

def SQLInsert(cardid, channel='A', t_now=None, status=1):
    if t_now is None:
        t_now = int(time.time()*1000)
        
    SQL_INSERT = '''
        INSERT INTO card_write_log(card_ID, timestamp, gate, success)
        VALUES('{card_ID}', {timestamp}, '{gate}', {success})
    '''.format(card_ID=cardid, timestamp=t_now, gate=channel, success=status)
    send_query(SQL_INSERT)


def getSQLcardID(shibboleth):
    SQL_QUERY = '''
        SELECT card_ID FROM user WHERE shibboleth="{shibboleth}"
    '''.format(shibboleth=shibboleth)
    query_result = send_query(SQL_QUERY)
    
    if len(query_result) == 0:
        return None
    
    return query_result[0]['card_ID']


def addbanner(img, banner):
    img[:banner.shape[0]] = banner
    return img


def smartbanner(img, nameCount=None):


    H, W, C = img.shape
    H = 50
    cv2.rectangle(img,(0,0),(W,H),(255,255,255),-1)
    
    if nameCount is not None:    
        for i, (n, c) in enumerate(name_counter[:3]):
            text = '%s (%2d)'%(n, c)
            
            cv2.putText(bgrImg, text, (30 + i*150, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=(0,0,0), 
                thickness=1, lineType=cv2.LINE_AA)

    else:
        text = 'No IDs to show...'
        cv2.putText(bgrImg, text, (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(0,0,0), 
                    thickness=1, lineType=cv2.LINE_AA)
        
                
    return img


if __name__ == '__main__':

    # Pre-check webcam before loading every other module
    cap = cv2.VideoCapture(0)
    ret, _ = cap.read()
    if not ret:
        raise RuntimeError('Video capture was unsuccessful.')


    AS = AsyncSaver(camID=0, rootPath='recordings/')

    if args.display:
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        if args.fullscreen:
            cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(
                'frame',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    if not args.virtual:
        pirate = ITKGatePirate()    
        
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('CUDA is used')
        cudnn.benchmark = True
    else:
        print('CUDA is not available')
    

    deploy_database = torch.load(args.database)
    names = deploy_database['names']
    cards = deploy_database['cards']
    embedding_vectors = deploy_database['embeddings'].half()
    print('Size of database: %5d samples' % len(embedding_vectors))     


    name_db = {name:i for i, name in enumerate(names) }
    card_db = {name:card for name, card in zip(names, cards)}    

    net = prepareOpenFace()
    net = net.eval()
    net.load_state_dict(torch.load(args.embedding_weights))

    if use_cuda:
        net.cuda()
        embedding_vectors = embedding_vectors.cuda()
    
    for p in net.parameters():
        p.requires_grad = False
    
    print('Model loaded')
    
    aligner = AlignDlib(args.dlib, region=args.region, grayScale=args.gray)    
    tensor_converter = ToTensor()

    bgrbanner = cv2.imread('banner.png')
    last_cardwrite = time.time()
    it = 0
    start_time = time.time()
    idle_begin = -1
    last_name = ''
    consecutive_occurrence = 0
    print('Begin capture')
    while True:
        it += 1
        try:
            # STEP 1: READ IMAGE
            ret, bgrImg = cap.read()
            bgrImg = cv2.flip(bgrImg, 1)
            if not ret:
                raise RuntimeError('Video capture was unsuccessful.')
                
            bb = aligner.getLargestFaceBoundingBox(bgrImg)
            if bb is None:
                if idle_begin < 0: 
                    idle_begin = time.time()
                idle_time = time.time() - idle_begin
                FPS = it / (time.time()-start_time)
                print('\t\t\tZzzzzz... No face detected (%4.0f sec), FPS:%2.2f\r' %\
                    (idle_time, FPS), flush=True, end='')
                
                if args.display:
                    if args.region:
                        # Draw region
                        
                        topleft = (aligner.regionXmin, aligner.regionYmin)
                        bottomright = (aligner.regionXmax, aligner.regionYmax)
                        cv2.rectangle(bgrImg, topleft, bottomright, (255, 255, 255), 3)
                        '''
                        cv2.putText(
                            bgrImg, 'Looking for a face...', 
                            (aligner.regionXmin, aligner.regionYmax+40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, (0, 0, 200), 1, cv2.LINE_AA)
                        '''
                    bgrImg = smartbanner(bgrImg)
                    cv2.imshow('frame', bgrImg)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break                
                continue
                
            idle_begin = -1
            AS.save(bgrImg, bb)
        

            # STEP 2: PREPROCESS IMAGE
            rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
            img = rgbImg
            aligned_img = aligner.align(96, img, bb=bb)
            
            #x = torch.FloatTensor(aligned_img).permute(2, 0, 1) / 255.
            x = tensor_converter(aligned_img)
            x = torch.autograd.Variable(x, volatile=True, requires_grad=False).detach()
            x = x[None]
            if use_cuda:
                x = x.cuda()
            
            # STEP 3: EMBEDD IMAGE
            inference_start = time.time()
            embedding128 = net(x)[0].half()
            inference_time = time.time() - inference_start            

           
            # STEP 4: COMPARE TO REGISTERED EMBEDDINGS
            topk_start = time.time()
            distances = ((embedding_vectors-embedding128.expand_as(embedding_vectors))**2).mean(-1)
            sorted_distances, idxs = torch.sort(distances)
            sorted_distances = sorted_distances[:args.k]
            idxs = idxs[:args.k]
            topk_time = time.time() - topk_start
            
            count_start = time.time()
            name_counter = {}
            for idx in idxs.data:
                n = names[idx]
                if name_counter.get(n) is None:
                    name_counter[n] = 1
                else:
                    name_counter[n] += 1
            name_counter = sorted(list(name_counter.items()), key=lambda x: x[1], reverse=True)[:args.k]
            count_time = time.time() - count_start
          
 
            # STEP 5: OPEN TURNSPIKE
            # TODO: design a good policy
            if name_counter[0][0].find('<') == -1 and name_counter[0][1]/args.k *100 > args.threshold and last_name == name_counter[0][0]:
                consecutive_occurrence += 1
                if consecutive_occurrence >= args.consecutive and (time.time() - last_cardwrite) > args.card_cooldown:
                    last_cardwrite = time.time()
                    card_id = getSQLcardID(last_name)
                    if card_id is not None:
                        print('OPEN:', last_name, card_id)
                        SQLInsert(card_id)
                        if not args.virtual:
                            pirate.emulateCardID(card_id)
                        
                        

            else:
                last_name = name_counter[0][0]
                consecutive_occurrence = 0
            
            # STEP 6: SHOW RESULTS
            #print('\x1b[2J')
            '''
            print('\tEmbedding network inference time: %1.4f sec' % inference_time)
            print('\tTop-k time: %1.4f sec' % topk_time)
            print('\tCount time: %1.4f sec' % count_time)

            print('consec', consecutive_occurrence, 'name', last_name)
            print('\t\t\tBENCHMARK WITH NAMES...\n')
            print('\t\t\t%20s:\t%4s:'%('name hash', 'occurrence'))
            '''
            
            for n, c in name_counter[:3]:
                print('\t\t\t%20s\t(%2d)'%(n.split()[-1], c))
            print('-'*80)
                
            '''
            FPS = it / (time.time()-start_time)
            print('\n\n\n')
            print('\t\t\tOpening soon! Stay tuned')
            print('\t\t\t  Info: sam.itk.ppke.hu\n\n')
            print('\tEmbedding network inference time: %1.4f sec, FPS=%2.2f' % (inference_time, FPS))
            '''
            # STEP 7: IF X IS AVAILABLE THEN SHOW FACE BOXES
            if args.display:
                (x, y, w, h) = rect_to_bb(bb)
                
                percentage = name_counter[0][1]/args.k*100
                x_offset = 40
                y_offset = 40
                radius_addition = 15
                font_scale = 1.5
                thickness = 2
                
                #color = (200, 200, 200)
                if percentage < args.threshold or name_counter[0][0].find('>') > -1:
                    color = (0, 0, 200)
                    text = '<UNK>'
    
                else: #consecutive_occurrence + args.consecutive / 3 > args.consecutive:
                    ratio = max(args.consecutive - consecutive_occurrence, 0) / args.consecutive
                    color = (ratio * 200, 200, ratio * 200)
                    text = '%s %2d %%'%(name_counter[0][0].split()[-1], percentage)
                
                
                cv2.putText(bgrImg, text, (x + x_offset-w//2, y + h + y_offset + radius_addition),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)
                
                circle_color = (255, 255, 255)
                circle_thickness = 1 
                if consecutive_occurrence >= args.consecutive:
                    circle_color = (0, 200, 0) 
                    circle_thickness = 5
                #cv2.rectangle(bgrImg, (bb.left(), bb.top()), (bb.right(), bb.bottom()), (0, 255, 0), 2)
                cv2.circle(bgrImg, (x+w//2, y+h//2), w//2+radius_addition, circle_color, circle_thickness)        

                
                bgrImg = smartbanner(bgrImg, name_counter)
                cv2.imshow('frame', bgrImg)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                


        except KeyboardInterrupt:
            print('Interrupted manually')
            break
            
            
            
