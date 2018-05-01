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


from utils import prepareOpenFace, AlignDlib, rect_to_bb
import gatepirate

containing_dir = str(pathlib.Path(__file__).resolve().parent)

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, 'weights')

parser = argparse.ArgumentParser()

parser.add_argument('--dlib', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(modelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--embedding-weights', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(modelDir, "openface.pth"))
parser.add_argument('--database', type=str, help='path to embedding2name database',
                    default=os.path.join(modelDir, "DEPLOY_DATABASE.tar"))
parser.add_argument('--k', type=int, help="List top K results", default=100)
parser.add_argument('--threshold', type=int, help="Threshold for opening count in %%", default=50)
parser.add_argument('--consecutive', type=int, help="How many frames is required to be authorized as the same person", 
                    default=30)
parser.add_argument('--ratio', type=float, help="Downsample input image", default=.8)
parser.add_argument('--display', action='store_true', help="Use OpenCV to show predictions on X")

args = parser.parse_args()

if __name__ == '__main__':
    if args.display:
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        pass
    #pirate = gatepirate.ITKGatePirate()    
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        print('CUDA is used')
        cudnn.benchmark = True
    else:
        print('CUDA is not available')
    

    deploy_database = torch.load(args.database)
    names = deploy_database['names']
    cards = deploy_database['cards']
    embedding_vectors = deploy_database['embeddings']
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
    
    aligner = AlignDlib(args.dlib)    
    tensor_converter = ToTensor()

    cap = cv2.VideoCapture(0)
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
            if not ret:
                raise RuntimeError('Video capture was unsuccessful.')
                
            rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
            img = cv2.resize(rgbImg, None, fx=args.ratio, fy=args.ratio)
            
            # STEP 2: PREPROCESS IMAGE
            bb = aligner.getLargestFaceBoundingBox(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
            if bb is None:
                if idle_begin < 0: 
                    idle_begin = time.time()
                idle_time = time.time() - idle_begin
                FPS = it / (time.time()-start_time)
                print('\t\t\tZzzzzz... No face detected (%4.0f sec), FPS:%2.2f\r' %\
                    (idle_time, FPS), flush=True, end='')
                
                if args.display:
                    cv2.imshow('frame', bgrImg)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break                
                continue
                
            idle_begin = -1
            aligned_img = aligner.align(96, img, bb=bb)
            
            #x = torch.FloatTensor(aligned_img).permute(2, 0, 1) / 255.
            x = tensor_converter(aligned_img)
            x = torch.autograd.Variable(x, volatile=True, requires_grad=False).detach()
            x = x[None]
            if use_cuda:
                x = x.cuda()
            
            # STEP 3: EMBEDD IMAGE
            inference_start = time.time()
            embedding128, _ = net(x)
            inference_time = time.time() - inference_start            

           
            # STEP 4: COMPARE TO REGISTERED EMBEDDINGS
            distances = ((embedding_vectors-embedding128.expand_as(embedding_vectors))**2).mean(-1)
            sorted_distances, idxs = torch.sort(distances)
            sorted_distances = sorted_distances[:args.k]
            idxs = idxs[:args.k]
            name_counter = {}
            for idx in idxs.data:
                n = names[idx]
                if name_counter.get(n) is None:
                    name_counter[n] = 1
                else:
                    name_counter[n] += 1
            name_counter = sorted(list(name_counter.items()), key=lambda x: x[1], reverse=True)[:args.k]
          
 
            # STEP 5: OPEN TURNSPIKE
            # TODO: design a good policy
            if name_counter[0][1]/args.k *100 > args.threshold and last_name == name_counter[0][0]:
                consecutive_occurrence += 1
                if consecutive_occurrence >= args.consecutive:
                    print('OPEN:', last_name, card_db[last_name])
                    #pirate.emulateCardID(card_db[name])
                    # Wait a few secs before continuing
	            #time.sleep(1.5)
            else:
                last_name = name_counter[0][0]
                consecutive_occurrence = 0
            
            # STEP 6: SHOW RESULTS
            print('\x1b[2J')
            print('consec', consecutive_occurrence, 'name', last_name)
            print('\t\t\tBENCHMARK WITH NAMES...\n')
            print('\t\t\t%20s:\t%4s:'%('name hash', 'occurrence'))
           
            for n, c in name_counter[:3]:
                print('\t\t\t%20s\t(%2d)'%(n.split()[-1], c))
            FPS = it / (time.time()-start_time)
            print('\n\n\n')
            print('\t\t\tOpening soon! Stay tuned')
            print('\t\t\t  Info: sam.itk.ppke.hu\n\n')
            print('\tEmbedding network inference time: %1.4f sec, FPS=%2.2f' % (inference_time, FPS))

            # STEP 7: IF X IS AVAILABLE THEN SHOW FACE BOXES
            if args.display:
                (x, y, w, h) = rect_to_bb(bb, args.ratio)
                
                percentage = name_counter[0][1]/args.k*100
                text = '%8s %2.1f %%'%(name_counter[0][0].split()[-1], percentage)
                x_offset = 0 
                y_offset = 40
                radius_addition = 15
                font_scale = 1.5
                thickness = 2
                
                #color = (200, 200, 200)
                if percentage < args.threshold or name_counter[0][0].find('>') > -1:
                    color = (0, 0, 200)
                else: #consecutive_occurrence + args.consecutive / 3 > args.consecutive:
                    ratio = max(args.consecutive - consecutive_occurrence, 0) / args.consecutive
                    print(ratio)
                    color = (ratio * 200, 200, ratio * 200)
                
                
                cv2.putText(bgrImg, text, (x + x_offset-w//2, y + h + y_offset + radius_addition),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)
                
                circle_color = (255, 255, 255)
                circle_thickness = 1 
                if consecutive_occurrence >= args.consecutive:
                    circle_color = (0, 200, 0) 
                    circle_thickness = 5
                
                cv2.circle(bgrImg, (x+w//2, y+h//2), w//2+radius_addition, circle_color, circle_thickness)

                
                cv2.imshow('frame', bgrImg)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                


        except KeyboardInterrupt:
            print('Interrupted manually')
            break
            
            
            
