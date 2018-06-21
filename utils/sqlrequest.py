import socket
'''
assert socket.gethostname() in ['k80','tegra-ubuntu','users'], \
    Exception('mySQL database can only be reached from k80, users or tegra')
'''

import pymysql
import time
import re
import cv2
from datetime import datetime

host = user = password = database = None

def initDB(conf_file='/home/botoscs/sam/utils/db.conf'):
    global host
    global user
    global password
    global database
    
    with open(conf_file) as f:
        host = f.readline().strip()
        user = f.readline().strip()
        password = f.readline().strip()
        database = f.readline().strip()
    
def db_query(query,args=None,as_dict=True,verbose=True):
    if host is None or user is None or password is None or database is None:
        raise RuntimeError('MySQL database is not initialised')    
    
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
    

'''
    CONVENIENCE FUNCTIONS
    mainly supporting the deployed face recognition script
'''

def push_images(photo_array, thumbnail_array, timestamps, card_IDs, verbose=False):
    def numpy2bytes(img):
        return cv2.imencode('.jpg', img)[1].tobytes()
    
    start = time.time()
    photos = [numpy2bytes(img) for img in photo_array]
    thumbnails = [numpy2bytes(img) for img in thumbnail_array]
    connection = pymysql.connect(host,user,password,database,charset='utf8mb4')
    try:
        with connection.cursor() as cursor:
            cursor.execute('SELECT Auto_increment FROM information_schema.tables WHERE table_name="photo"')
            first_ID = cursor.fetchone()[0] + 1
            photo_IDs = list(range(first_ID, first_ID+len(photos)))
            query = 'INSERT INTO photo (photo_ID,full_frame,timestamp) VALUES (%s,%s,%s)'
            cursor.executemany(query, zip(photo_IDs, photos, timestamps))
            
            cursor.execute('SELECT Auto_increment FROM information_schema.tables WHERE table_name="thumbnail"')
            first_ID = cursor.fetchone()[0] + 1
            thumbnail_IDs = list(range(first_ID, first_ID+len(photos)))
            query = 'INSERT INTO thumbnail (thumbnail_ID,photo_ID,thumbnail_img) VALUES (%s,%s,%s)'
            cursor.executemany(query, zip(thumbnail_IDs, photo_IDs, thumbnails))
            
            query = 'INSERT INTO annotation (thumbnail_ID,card_ID) VALUES (%s,%s)'
            cursor.executemany(query, zip(thumbnail_IDs, card_IDs))
            
            connection.commit()
    finally:
        connection.close()
    end = time.time()

    if verbose:
        time_str = time.strftime('%m/%d/%Y %H:%M:%S', time.gmtime(end))
        print('REQUEST SUCCESS!')
        print('Query took %3.2f sec' % (end-start))



def getName(card_ID):
    '''
        MySQL helper function for single queries
        
        Args: card_ID
    
        Returns: the corresponding name to card_ID, 
          if card_ID is not registered it returns None
    '''
    
    SQL_QUERY = '''
        SELECT name FROM user WHERE card_ID="{card_ID}"
    '''.format(card_ID=card_ID)
    query_result = db_query(SQL_QUERY, verbose=False)
    
    if len(query_result) == 0:
        return None
    
    return query_result[0]['name']
    


def getCard2Name():
    '''
        MySQL helper function for updating whole {card_ID: name} dict
        
        Returns: a whole new dictionary of the current known card_ID <-> name pairs
    '''
    qresult = db_query('SELECT shibboleth, card_ID FROM user')
    card2name = {q['card_ID']:q['shibboleth'] for q in qresult}
    return card2name
