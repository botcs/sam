import socket
import pymysql
import cv2
import numpy as np
import threading
import time

with open('/home/botoscs/sam/utils/db.conf','r') as f:
    host = f.readline().strip()
    user = f.readline().strip()
    password = f.readline().strip()
    database = f.readline().strip()
    
class AsyncSQLUploader:
    def __init__(self, bufferSize=20):
        self.bufferSize = bufferSize
        self.emptyBuffer()
        self.locked = False
        threading.Thread(target=self.flushCheck).start()

    def emptyBuffer(self):
        self.photos = []
        self.thumbnails = []
        self.timestamps = []
        self.card_IDs = []
        self.xmins = []
        self.ymins = []
        self.xmaxs = []
        self.ymaxs = []
        self.discardCounter = 0

        
    def numpy2bytes(self,img):
        if img.shape[0] == 0 or img.shape[1] == 0:
            return False
        return cv2.imencode('.jpg', img)[1].tobytes()
        
    def add_single(self,photo,timestamp,card_ID,BB):
        self.discardCounter += 1
        if self.discardCounter < 3:
            return
        self.discardCounter = 0
        assert type(photo) is np.ndarray, "type of photo must be numpy.ndarray"
        assert type(timestamp) is float, "type of timestamp must be float"
        assert type(card_ID) is str, "type of card_ID must be string"
        assert type(BB) is tuple, "Bounding Boxes are (xmin, ymin, xmax, ymax) touples"
        xmin,ymin,xmax,ymax = BB
        assert type(xmin) is int, "type of xmin must be int"
        assert type(ymin) is int, "type of ymin must be int"
        assert type(xmax) is int, "type of xmax must be int"
        assert type(ymax) is int, "type of ymax must be int"
        
        self.photos.append(self.numpy2bytes(photo))
        self.thumbnails.append(self.numpy2bytes(photo[ymin:ymax,xmin:xmax]))
        self.timestamps.append(int(timestamp*1000))
        self.card_IDs.append(card_ID)
        self.xmins.append(xmin)
        self.ymins.append(ymin)
        self.xmaxs.append(xmax)
        self.ymaxs.append(ymax)
        
        #if len(self.photos) > self.bufferSize:
            #self.flush()
            #if not self.locked:
            #    self.locked = True
            #    threading.Thread(target=self.flush).start()
            
    def add_multi(self,photos,timestamps,card_IDs,BBs):
        #print(list(map(type, [photos, timestamps, card_IDs, BBs])))
        assert type(photos) is list and type(timestamps) is list and type(card_IDs) is list and \
               type(BBs) is list, \
            "all parameters must be lists"
        xmins, ymins, xmaxs, ymaxs = zip(*BBs)
        self.photos += [self.numpy2bytes(img) for img in photos]
        self.thumbnails += [self.numpy2bytes(img[max(0,ymins[i]):ymaxs[i],min(0,xmins[i]):xmaxs[i]]) for i,img in enumerate(photos)]
        self.timestamps += [int(t*1000) for t in timestamps]
        self.card_IDs += card_IDs
        self.xmins += xmins
        self.ymins += ymins
        self.xmaxs += xmaxs
        self.ymaxs += ymaxs
        
        #if len(self.photos) > self.bufferSize:
            #self.flush()
            #if not self.locked:
            #    self.locked = True
            #    threading.Thread(target=self.flush).start()
            
    def flushCheck(self):
        while(True):
            #start = time.time()
            if len(self.photos) > self.bufferSize:
                self.flush()
            time.sleep(.3)
            #end = time.time()
            #if (3 - (end-start) > 0):
                #time.sleep(3 - (end-start))
        
        
    def flush(self):
        if len(self.photos) == 0:
            self.locked = False
            return
        connection = pymysql.connect(host,user,password,database,charset='utf8mb4')
        try:
            with connection.cursor() as cursor:
                cursor.execute('SELECT Auto_increment FROM information_schema.tables WHERE table_name="photo"')
                first_ID = cursor.fetchone()[0] + 1
                photo_IDs = list(range(first_ID, first_ID+len(self.photos)))
                query = 'INSERT INTO photo (photo_ID,photo_img,timestamp) VALUES (%s,%s,%s)'
                cursor.executemany(query,zip(photo_IDs,self.photos,self.timestamps))
                
                for index,value in enumerate(self.thumbnails):
                    if (not value):
                        del self.thumbnails[index]
                        del photo_IDs[index]
                        del self.xmins[index]
                        del self.ymins[index]
                        del self.xmaxs[index]
                        del self.ymaxs[index]
                        del self.card_IDs[index]

                cursor.execute('SELECT Auto_increment FROM information_schema.tables WHERE table_name="thumbnail"')
                first_ID = cursor.fetchone()[0] + 1
                thumbnail_IDs = list(range(first_ID, first_ID+len(self.thumbnails)))
                query = 'INSERT INTO thumbnail (thumbnail_ID,photo_ID,thumbnail_img,xmin,ymin,xmax,ymax) VALUES (%s,%s,%s,%s,%s,%s,%s)'
                cursor.executemany(query, zip(thumbnail_IDs,photo_IDs,self.thumbnails,
                                              self.xmins,self.ymins,self.xmaxs,self.ymaxs))
                
                
                #print(thumbnail_IDs)
                query = 'INSERT INTO annotation (thumbnail_ID,card_ID) VALUES (%s,%s)'
                cursor.executemany(query, zip(thumbnail_IDs,self.card_IDs))

                connection.commit()
        finally:
            connection.close()
        
        self.emptyBuffer()
        self.locked = False
