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
