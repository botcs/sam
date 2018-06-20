import time
import torch
from torch.cuda import FloatTensor as Tensor
from .async_sql_uploader import AsyncSQLUploader

class Tracer:
    '''
        Traces the bounding box(es) appearing on the stream
        
        once the ID is revealed of the traced box (i.e. by reading the card) 
        then all previous bounding box are tagged as well retrospectively
        
        
        When AUTH occurs then assign latest frame ONLINE
        
        When tracing session dies -> bbox x coord jumps further than treshold then
         assign all previously unassigned 
    '''
    def __init__(self, x_displacement_treshold=100, SQLBufferSize=30):
        self.treshold = x_displacement_treshold
        self.last_xmin = None
        self.bbox_distances = [0]
        self.uploader = AsyncSQLUploader(SQLBufferSize)
        self.initCache()
                
    def initCache(self):
        self.cached_embeddings = Tensor(0, 128)
        self.cached_fullframes = []
        self.cached_bboxes = []
        self.cached_IDs = []
        self.cached_times = []
    
    
    def track(self, bgrImg, mainBB, embedding128, AUTHORIZED_ID, KNOWN_DB, virtual=False):
        xmin, ymin, xmax, ymax = mainBB.left(), mainBB.top(), mainBB.right(), mainBB.bottom()
    
        if self.last_xmin is None:
            self.last_xmin = mainBB.left()
        else:
            self.bbox_distances.append(abs(self.last_xmin - xmin))
            self.last_xmin = mainBB.left()
        
        if self.bbox_distances[-1] < self.treshold:
            
            if AUTHORIZED_ID is None:
                # BEGIN CACHE-ING UNKNOWN SAMPLES
                self.cached_embeddings = torch.cat([self.cached_embeddings, embedding128])
                self.cached_fullframes.append(bgrImg.copy())
                self.cached_bboxes.append((xmin, ymin, xmax, ymax))
                self.cached_IDs.append(AUTHORIZED_ID)
                self.cached_times.append(time.time())
            else:
                if len(self.cached_embeddings) > 0:
                    print('RETROGRADE ASSIGNMENT: "%s" of %d images'%(AUTHORIZED_ID, len(self.cached_embeddings)))
                    KNOWN_DB['emb'] = torch.cat([KNOWN_DB['emb'], self.cached_embeddings])
                    KNOWN_DB['id'].extend([AUTHORIZED_ID for _ in range(len(self.cached_embeddings))])
                    if True:#not virtual:
                        self.uploader.add_multi(
                            photos=self.cached_fullframes, 
                            timestamps=self.cached_times, 
                            card_IDs=self.cached_IDs, 
                            BBs=self.cached_bboxes)
                    
                    self.initCache()
                
                print('ONLINE ASSIGNMENT: "%s"'%AUTHORIZED_ID)
                KNOWN_DB['emb'] = torch.cat([KNOWN_DB['emb'], embedding128])
                KNOWN_DB['id'].append(AUTHORIZED_ID)
                if True:#not virtual:
                    self.uploader.add_single(
                        bgrImg, time.time(), AUTHORIZED_ID, (xmin, ymin, xmax, ymax))
                
        else: 
            self.initCache()
            AUTHORIZED_ID = None
        
        return AUTHORIZED_ID, KNOWN_DB
