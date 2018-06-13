import socket
assert socket.gethostname() in ['k80','tegra-ubuntu','users'], \
    Exception('mySQL database can only be reached from k80, users or tegra')

import pymysql
import time
import re
from datetime import datetime

with open('/home/csbotos/sam/utils/db.conf') as f:
    host = f.readline().strip()
    user = f.readline().strip()
    password = f.readline().strip()
    database = f.readline().strip()
    
def db_query(query,args=None,as_dict=True,verbose=True):
    start = time.time()
    if verbose:
        time_str = datetime.fromtimestamp(start).strftime('%m/%d/%Y %H:%M:%S')
        print('QUERY SENT - ', time_str)
        print('QUERY = """\n', query, '\n"""')
        
    connection = pymysql.connect(host,user,password,database,charset='utf8mb4',
                   cursorclass=pymysql.cursors.DictCursor if as_dict else pymysql.cursors.Cursor)
    try:
        with connection.cursor() as cursor:
            if args:
                assert type(args) in [tuple, list], 'parameter "args" must be tuple or list'
                if type(args[0]) in [tuple, list, dict]:
                    cursor.executemany(query,args)
                else:
                    cursor.execute(query,args)
            else:
                cursor.execute(query)
                
            if query.strip().lower().startswith('select'):
                result = cursor.fetchall()
            else:
                connection.commit()
                result = None
    finally:
        connection.close()
    end = time.time()

    if verbose:
        time_str = time.strftime('%m/%d/%Y %H:%M:%S', time.gmtime(end))
        
        print('REQUEST SUCCESS!')
        if result:
            print('Length of result:', len(result))
    
    print('Query took %3.2f sec' % (end-start))
    return result

def push_images(photo_array,thumbnail_array,timestamps,IDs):
    def numpy2bytes(img):
        return cv2.imencode('.jpg', img)[1].tobytes()
    
    start = time.time()
    photos = [numpy2bytes(img) for img in photo_array]
    thumbnails = [numpy2bytes(img) for img in humbnail_array]
    connection = pymysql.connect(host,user,password,database,charset='utf8mb4')
    try:
        with connection.cursor() as cursor:
            cursor.execute('SELECT Auto_increment FROM information_schema.tables WHERE table_name=photo')
            first_ID = cursor.fetch()[0] + 1
            photo_IDs = list(range(first_ID, first_ID+len(photos)))
            query = 'INSERT INTO photo (photo_ID,full_frame,timestamp) VALUES (%s,%s,%s)'
            cursor.executemany(query,zip(photo_IDs,photos,timestamps))
            
            cursor.execute('SELECT Auto_increment FROM information_schema.tables WHERE table_name=thumbnail')
            first_ID = cursor.fetch()[0] + 1
            thumbnail_IDs = list(range(first_ID, first_ID+len(photos)))
            query = 'INSERT INTO thumbnail (thumbnail_ID,photo_ID,thumbnail_img,timestamp) VALUES (%s,%s,%s,%s)'
            cursor.executemany(query,zip(thumbnail_IDs,photo_IDs,photos,timestamps))
            
            query = 'INSERT INTO annotation (thumbnail_ID,card_ID) VALUES (%s,%s)'
            cursor.executemany(query,zip(thumbnail_IDs,card_ID))
            
            connection.commit()
    finally:
        connection.close()
    end = time.time()

    if verbose:
        time_str = time.strftime('%m/%d/%Y %H:%M:%S', time.gmtime(end))
        print('REQUEST SUCCESS!')
        print('Query took %3.2f sec' % (end-start))
