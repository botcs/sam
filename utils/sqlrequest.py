from urllib import request, parse
import json
import time
import re
from datetime import datetime
import sys

with open('/home/csbotos/sam/utils/db-password') as dbpf:
    dbp = dbpf.readline()

def send_query(query, verbose=True):
    start = time.time()
    if verbose:
        time_str = datetime.fromtimestamp(start).strftime('%m/%d/%Y %H:%M:%S')
        print('QUERY SENT - ', time_str)
        print('QUERY = """\n', query, '\n"""')
        
    post_data = {'password': dbp, 
                 'query': query.replace('\n', ' ') }
    data = parse.urlencode(post_data).encode()
    req =  request.Request("https://users.itk.ppke.hu/~hakta/belepteto/db.php", data=data)
    resp = request.urlopen(req)
    response = json.loads(resp.read().decode())
    end = time.time()
    
    assert response['success'], response['answer']
    if verbose:
        time_str = time.strftime('%m/%d/%Y %H:%M:%S', time.gmtime(end))
        
        print('REQUEST SUCCESS!')
        print('Length of result:', len(response['answer']))
    
    print('Query took %3.2f sec' % (end-start))
    return response['answer']

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
