import socket
import pymysql
import cv2
import numpy as np
import threading
import time
from PIL import Image
from io import BytesIO
from .sqlrequest import initDB


class AsyncSQLUploader:
    def __init__(self, bufferSize=20):
        self.dbCredentials = initDB()
        
        self.bufferSize = bufferSize
        self.emptyBuffer()
        self.locked = False
        self.discardNum = 0
        self.lastFlushTime = -1

    def emptyBuffer(self):
        self.timestamps = []
        self.full_frames = []
        self.BBs = []
        self.card_IDs = []
        self.is_preds = []
        self.discardCounter = 0

    def numpy2bytes(self, bgrImg):
        img = Image.fromarray(bgrImg[:,:,::-1])
        output = BytesIO()
        #img.save(output, format='JPEG', quality=85, progressive=True, optimize=True)
        img.save(output, format='JPEG', quality=80, progressive=False, optimize=False)
        contentJPG = output.getvalue()
        #print('JPG:', len(contentJPG))
        output.close()
        return contentJPG
        #return cv2.imencode('.jpg', img)[1].tobytes()

    def add_single(self, photo, timestamp, card_ID, BB, is_pred=False):
        self.discardCounter += 1
        if self.discardCounter < self.discardNum:
            return
        self.discardCounter = 0
        self.timestamps.append(int(timestamp*1000))
        self.full_frames.append(photo)
        self.card_IDs.append(card_ID)
        self.BBs.append(BB)
        self.is_preds.append(is_pred)
        self.flushCheck()

    def add_multi(self, photos, timestamps, card_IDs, BBs, is_pred=False):
        self.full_frames += photos
        self.timestamps += [int(t*1000) for t in timestamps]
        self.card_IDs += card_IDs
        self.BBs += BBs
        self.is_preds += [is_pred for _ in range(len(photos))]
        self.flushCheck()

    def cropThumbnail(self, img, BB, paddingRatio=0.35):
        xmin, ymin, xmax, ymax = BB
        H, W, C = img.shape
        bbH = xmax - xmin
        bbW = ymax - ymin

        xminPadded = max(0, xmin - int(bbW * paddingRatio))
        yminPadded = max(0, ymin - int(bbH * paddingRatio))
        xmaxPadded = min(W, xmax + int(bbW * paddingRatio))
        ymaxPadded = min(H, ymax + int(bbH * paddingRatio))

        cropped_img = img[yminPadded:ymaxPadded, xminPadded:xmaxPadded]
        #resized_img = cv2.resize(cropped_img, (min(125, bbW), min(125, bbH)))

        return cropped_img

    def flushCheck(self):
        if len(self.full_frames) > self.bufferSize and not self.locked:

            self.locked = True
            threading.Thread(target=self.flush).start()

    def flush(self, flushBufferSizeOnly=True):
        flush_start = time.time()
        connection = pymysql.connect(*self.dbCredentials,charset='utf8mb4')
        try:
            if flushBufferSizeOnly:
                flush_full_frames = self.full_frames[-self.bufferSize:]
                flush_timestamps = self.timestamps[-self.bufferSize:]
                flush_card_IDs = self.card_IDs[-self.bufferSize:]
                flush_BBs = self.BBs[-self.bufferSize:]
                flush_is_preds = self.is_preds[-self.bufferSize:]

                self.full_frames = self.full_frames[:-self.bufferSize]
                self.timestamps = self.timestamps[:-self.bufferSize]
                self.card_IDs = self.card_IDs[:-self.bufferSize]
                self.BBs = self.BBs[:-self.bufferSize]
                self.is_preds = self.is_preds[:-self.bufferSize]
            else:
                flush_full_frames = self.full_frames
                flush_timestamps = self.timestamps
                flush_card_IDs = self.card_IDs
                flush_BBs = self.BBs
                flush_is_preds = self.is_preds

            with connection.cursor() as cursor:
                full_frames = list(map(self.numpy2bytes,flush_full_frames))
                thumbnails = [self.cropThumbnail(img, BB)
                    for img, BB in zip(flush_full_frames, flush_BBs)]
                thumbnails = list(map(self.numpy2bytes,thumbnails))
                xmins,ymins,xmaxs,ymaxs = zip(*flush_BBs)
                # Bool converts to 0, 1
                flush_is_it_sures = [0 if is_pred else 1 for is_pred in flush_is_preds]

                query = '''
                    INSERT INTO photo (timestamp,full_frame,thumbnail,xmin,ymin,xmax,ymax,card_ID,is_it_sure)
                    VALUES            (       %s,        %s,       %s,  %s,  %s,  %s,  %s,     %s,        %s)
                '''
                cursor.executemany(
                    query,
                    zip(flush_timestamps,
                        full_frames,
                        thumbnails,
                        xmins,ymins,xmaxs,ymaxs,
                        flush_card_IDs,
                        flush_is_it_sures)
                    )
                connection.commit()

            stat_str = 'Flush took %3.4f seconds' % (time.time()-flush_start)
            if not flushBufferSizeOnly:
                self.emptyBuffer()
                print(stat_str)
            else:
                print(stat_str + ', %d entries remained in the buffer'%len(self.BBs))

        finally:
            connection.close()
            self.locked = False
