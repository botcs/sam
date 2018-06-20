# Echo client program
import socket
import time
import pickle

# Test Data
data = 'whatever'


import numpy as np
data = np.random.randn(3, 480, 640)

import dlib
data = dlib.rectangle(*[0, 0, 10, 100])


start_time = time.time()
times = []
HOST = '10.3.19.208'    # The remote host
PORT = 50007              # The same port as used by the server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
for it in range(10000):
    t = time.time()
    serialized_data = pickle.dumps(data)
    s.send(serialized_data)
    data = pickle.loads(s.recv(4096))
    times.append(time.time() - t)
    #print(times[-1])
s.close()
print('Received', repr(data))
print('Total time:', time.time()-start_time) 
print('min/avg/max = %3.3f/%3.3f/%3.3f' % (min(times), sum(times)/len(times), max(times)))
