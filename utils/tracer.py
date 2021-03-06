import time
import torch
from torch.cuda import FloatTensor as Tensor
from .async_sql_uploader import AsyncSQLUploader

class Tracer:
    '''
        ABSTRACT CLASS
        Traces the bounding box(es) appearing on the stream
        
        once the ID is revealed of the traced box (i.e. by reading the card) 
        then all previous bounding box are tagged as well retrospectively
         
        
        When AUTH occurs then assign latest frame ONLINE
        
        When tracing session dies -> bbox x coord jumps further than treshold then
         assign all previously unassigned 
    '''
    def __init__(self, x_displacement_treshold=40, SQLBufferSize=5):
        self.treshold = x_displacement_treshold
        self.last_xmin = None
        self.bbox_distances = [0]
        self.uploader = AsyncSQLUploader(SQLBufferSize)
        self.traced_id = None
        self.initCache()
                
    def initCache(self):
        self.cached_embeddings = Tensor(0, 128)
        self.cached_fullframes = []
        self.cached_bboxes = []
        self.cached_times = []
    
    
    def flush(self):
        self.uploader.flushCheck()
    
class CardValidationTracer(Tracer):            
    '''
    Implements policy where:
        - Caching BEGINS when a face enters the screen
        - The cache is UPLOADED to the MySQL server when card AUTH happens
        - If horizontal displacement of the face is larger than the treshold
          the cache is EMPTIED
    '''
    
    def track(self, bgrImg, mainBB, embedding128, AUTHORIZED_ID, KNOWN_DB, virtual=False):
        xmin, ymin, xmax, ymax = mainBB.left(), mainBB.top(), mainBB.right(), mainBB.bottom()
    
        if self.last_xmin is None:
            self.last_xmin = mainBB.left()
        else:
            self.bbox_distances.append(abs(self.last_xmin - xmin))
            self.last_xmin = mainBB.left()
        
        if self.bbox_distances[-1] < self.treshold:
            
            if AUTHORIZED_ID is None and self.traced_id is None:
                # BEGIN CACHE-ING UNKNOWN SAMPLES
                self.cached_embeddings = torch.cat([self.cached_embeddings, embedding128])
                self.cached_fullframes.append(bgrImg)
                self.cached_bboxes.append((xmin, ymin, xmax, ymax))
                self.cached_times.append(time.time())
            else:
                if self.traced_id is None:            
                    self.traced_id = AUTHORIZED_ID
                    print('RETROGRADE ASSIGNMENT: "%s" of %d images'%(AUTHORIZED_ID, len(self.cached_embeddings)))
                    KNOWN_DB['emb'] = torch.cat([KNOWN_DB['emb'], self.cached_embeddings])
                    KNOWN_DB['id'].extend(
                        [AUTHORIZED_ID for _ in range(len(self.cached_embeddings))])
                    if True:#not virtual:
                        t = time.time()
                        self.uploader.add_multi(
                            photos=self.cached_fullframes, 
                            timestamps=self.cached_times, 
                            card_IDs=[AUTHORIZED_ID for _ in range(len(self.cached_embeddings))], 
                            BBs=self.cached_bboxes)
                        
                    self.initCache()
                else:
                    #print('TRACED ASSIGNMENT: "%s"'%self.traced_id)
                    KNOWN_DB['emb'] = torch.cat([KNOWN_DB['emb'], embedding128])
                    KNOWN_DB['id'].append(self.traced_id)
                    if True:#not virtual:
                        t = time.time()
                        self.uploader.add_single(
                            bgrImg, t, self.traced_id, (xmin, ymin, xmax, ymax))

                        with open('async-time-plot.txt', 'a') as f:
                            f.write('%1.9f\n'%(time.time()-t))
                
        else: 
            self.initCache()
            self.traced_id = None
        
        
        return KNOWN_DB
        
        
        
        
class PredictionTracer(Tracer):            
    '''
    Implements policy where:
        - Caching BEGINS when a face is RECOGNIZED (see the main recog. policy)
        - The cache is UPLOADED to the MySQL server when face leaves the frame
        - If horizontal displacement of the face is larger than the treshold
          the cache is EMPTIED
    '''
        
    def addPrediction(self, bgrImg, mainBB, RECOGNIZED_ID):
        xmin, ymin, xmax, ymax = mainBB.left(), mainBB.top(), mainBB.right(), mainBB.bottom()
    
        self.uploader.add_single(
            bgrImg, 
            time.time(), 
            RECOGNIZED_ID, 
            (xmin, ymin, xmax, ymax),
            is_pred=True
        )

