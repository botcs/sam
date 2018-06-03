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
    
def send_query(query, verbose=True):
    start = time.time()
    if verbose:
        time_str = datetime.fromtimestamp(start).strftime('%m/%d/%Y %H:%M:%S')
        print('QUERY SENT - ', time_str)
        print('QUERY = """\n', query, '\n"""')
        
    db = pymysql.connect(host,user,password,database)
    try:
        with pymysql.cursors.DictCursor(db) as cursor:
            cursor.execute(query)
            db.commit()
            result = cursor.fetchall()
    finally:
        db.close()
    end = time.time()

    if verbose:
        time_str = time.strftime('%m/%d/%Y %H:%M:%S', time.gmtime(end))
        
        print('REQUEST SUCCESS!')
        print('Length of result:', len(result))
    
    print('Query took %3.2f sec' % (end-start))
    return result

def send_large_query(query, batch_size=100000, verbose=True): 
    counter_SQL = re.sub(r'SELECT (.*) FROM', 'SELECT count(*) FROM', query) 
    result_length = int(send_query(counter_SQL)[0]['count(*)']) 
    print("Expected length of result:", result_length)
    result = [] 
    for i in range(0, result_length, batch_size): 
        result += send_query(query + " LIMIT {}, {}".format(i, batch_size), verbose)
        if verbose:
            print("Recieved chunk: {} - {}".format(
                i, i+batch_size if i+batch_size < result_length else result_length)) 
    
    return result