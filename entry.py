import utils
import os
import time
import argparse
import pathlib
import torch
import cv2
import torch.backends.cudnn as cudnn
from torch.autograd import Variable


from utils import prepareOpenFace, AlignDlib
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
parser.add_argument('--k', type=int, help="List top K results", default=7)

args = parser.parse_args()

if __name__ == '__main__':
    pirate = gatepirate.ITKGatePirate()    
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
    cap = cv2.VideoCapture(0)
    it = 0
    start_time = time.time()
    idle_begin = -1
    print('Begin capture')
    while True:
        it += 1
        try:
            # STEP 1: READ IMAGE
            ret, bgrImg = cap.read()
            if not ret:
                raise RuntimeError('Video capture was unsuccessful.')
                
            rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
            img = rgbImg
            
            # STEP 2: PREPROCESS IMAGE
            aligned_img = aligner.align(96, img)
            if aligned_img is None:
                if idle_begin < 0: 
                    idle_begin = time.time()
                idle_time = time.time() - idle_begin
                print('\t\t\tZzzzzz... No face detected (%4.0f sec)\r' % idle_time, flush=True, end='')

                continue
            idle_begin = -1
            x = torch.FloatTensor(aligned_img).permute(2, 0, 1)
            x = torch.autograd.Variable(x, volatile=True, requires_grad=False)
            x = x[None]
            if use_cuda:
                x = x.cuda()
            
            # STEP 3: EMBEDD IMAGE
            inference_start = time.time()
            embedding128, _ = net(x)
            inference_time = time.time() - inference_start            

           
            # STEP 4: COMPARE TO REGISTERED EMBEDDINGS
            distances = ((embedding_vectors-embedding128.expand_as(embedding_vectors))**2).mean(-1)
            idxs = torch.sort(distances)[1][:200]
            name_counter = {}
            for idx in idxs.data:
                n = names[idx]
                if name_counter.get(n) is None:
                    name_counter[n] = 1
                else:
                    name_counter[n] += 1
            name_counter = sorted(list(name_counter.items()), key=lambda x: x[1], reverse=True)[:args.k]
          
 
            # STEP 5: OPEN TURNSPIKE
            for name, _ in name_counter[:2]:
                if len(name) > 0:
                    print('OPEN:', name, card_db[name])
                    pirate.emulateCardID(card_db[name])
                    # Wait a few secs before continuing
                    time.sleep(1.5)
                    break
                else:
                    pass
                    #print('Negative sample found')

            # STEP 6: SHOW RESULTS
            #print('\x1b[2J')
            print('\t\t\tBENCHMARK WITHOUT NAMES...\n')
            print('\t\t\t%9s:\t%4s:'%('name hash', 'occurrence'))
            for n, c in name_counter:
                print('\t\t\t%4s\t\t(%2d)'%(n, c))
                #print('\t\tUser hash: %15s\tdistance: %0.6f' % (str(abs(hash(names[idx])))[:4], distances[idx]))    
                #print('\t\tUser : %s  distance: %0.6f' % (names[idx], distances[idx]))    
           
            FPS = it / (time.time()-start_time)
            print('\n\n\n')
            print('\t\t\tOpening soon! Stay tuned')
            print('\t\t\t  Info: sam.itk.ppke.hu\n\n')
            print('\tEmbedding network inference time: %1.4f sec, FPS=%2.2f' % (inference_time, FPS))

        except KeyboardInterrupt:
            print('Interrupted manually')
            break
            
            
            
