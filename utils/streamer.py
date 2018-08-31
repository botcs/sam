import socket
import struct
import time
import select
import threading
import warnings

class Messenger(object):
    """
    Wraps a socket: implements the reading and writing with some conveniencies
      - reads the whole message only
      - (Optional) discard messages which are older than specified time-constant
      - (Optional) discard messages which would break the order of sending
    """

    def __init__(self, socket, discard_older=None, only_consecutive=False):
        self.discard_older = discard_older
        self.only_consecutive = only_consecutive
        self.sock = socket
        if only_consecutive:
            self.last_timestamp = -1
    
    def send_msg(self, msg):        
        # Prefix each message with:
        # - its length '>I' -> big endian unsigned integer 32bits = 4 bytes
        # - current timestamp '>Q' -> big endian unsigned long 8 bytes
        # Overall message length 4+4+N bytes
        sock = self.sock
        msglen = len(msg)
        msgtime = int(time.time()*1000)
        msg = struct.pack('>I', msglen) + struct.pack('>Q', msgtime) + msg
        sock.sendall(msg)

    def recv_msg(self):
        # Read message length and unpack it into an integer
        sock = self.sock
        raw_msglen = self.recvall(4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>I', raw_msglen)[0]
        
        raw_msgtime = self.recvall(8)
        if not raw_msgtime:
            return None
        msgtime = struct.unpack('>Q', raw_msgtime)[0] / 1000
        
        if self.discard_older is not None:
            current_time = time.time()
            delay = current_time - msgtime
            if delay < 0:
                warnings.warn('Delay time is negative (%3f sec), sender-receiever time.time() function may be out of sync'%delay, RuntimeWarning)
            if delay > self.discard_older:
                return None
        
        if self.only_consecutive:
            if msgtime < self.last_timestamp:
                return None
            self.last_timestamp = msgtime
        # Read the message data
        return self.recvall(msglen)

    def recvall(self, n):
        # Helper function to recv n bytes or return None if EOF is hit
        sock = self.sock
        data = b''
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data


class StreamerAbstract(threading.Thread):
    
    def __init__(self, address, 
                 max_retries=1000, 
                 discard_older=None, 
                 only_consecutive=False,
                 connection_timeout=.5):
                 
        super(StreamerAbstract, self).__init__()
        self.address = address
        self.timeout = connection_timeout
        
        self.read_queue = []
        self.write_queue = []
        
        self.retries = 0
        self.max_retries = max_retries
        
        self.discard_older = discard_older
        self.only_consecutive = only_consecutive
        self.running = True
        self.start()
        
    def emptyBuffer(self):
        self.emptyWriteBuffer()
        self.emptyReadBuffer()
    
    
    def emptyWriteBuffer(self):
        self.write_queue = []
        
    
    def emptyReadBuffer(self):
        self.read_queue = []
        
        
    def stop(self):
        self.join()
        
        
    def close():
        self.join()
        
            
    def join(self):
        self.running = False
        print("Stopping Streamer...")
        # Block until the RUN() ends itself
        super(StreamerAbstract, self).join()

        
    def connectSocket(self, socket):
        assert False, "Must be implemented in derived class"
        
    def run(self):
        sock_addr = self.address
        sock = None
        listener = self.getSocket()
        while self.retries < self.max_retries and self.running:
            try:                          
                self.retries += 1              
                print('Connecting to server... [%4d/%4d]'%
                      (self.retries, self.max_retries))
                sock = self.connectSocket(listener)
                print('Connection established: %s:%d'%sock_addr)
                
                messenger = Messenger(
                    sock, self.discard_older, self.only_consecutive)
                    
                while self.running:
                    selection = select.select([sock], [sock], [sock], 1)
                    ready_to_read, ready_to_write, in_error = selection
                    
                    try:
                        if len(ready_to_read) == 1:
                            msg = messenger.recv_msg()
                            if msg is None:
                                raise ConnectionAbortedError('Connection Aborted')
                            lock = threading.Lock()
                            lock.acquire()
                            self.read_queue.append(msg)
                            lock.release()                    
                            
                            self.retries = 0
                            
                        if len(ready_to_write) == 1 and len(self.write_queue) > 0:
                            lock = threading.Lock()
                            lock.acquire()
                            msg = self.write_queue.pop(0)
                            lock.release()
                            messenger.send_msg(msg)

                            self.retries = 0
                        
                    except (BrokenPipeError, ConnectionResetError) as e:
                        print('Connection broke up: %s:%d'%sock_addr,
                              'HANDLED error: ', e, end='\t')
                        continue
                    # Just in case this monitoring would overwhelm the OS
                    # I know... there must be something cooler
                    #time.sleep(.05)
                    
            except ConnectionRefusedError as e:
                print('Unsuccesful connection: %s:%d'%sock_addr,
                      'HANDLED error: ', e, end='\t')
            except ConnectionAbortedError as e:
                sock.close()
                print('Connection broke up: %s:%d'%sock_addr,
                      'HANDLED error: ', e, end='\t')
                continue

            except socket.timeout:
                print('Connection timeout', end='\t')

            except KeyboardInterrupt:
                print('/nManual interruption')
                print('Connection broke up: %s:%d'%sock_addr, end='\t')
                self.running = False
                
            finally:
                if self.retries >= self.max_retries:
                    print('max_retries exceeded, exiting stream loop...')
                elif not self.running:
                    print('.running set to False, exiting stream loop...')
                else:
                    print('retry...')
                    time.sleep(self.timeout)

            
        self.running = False
        if listener is not None:
            listener.close()
            print('listener closed')
        if sock is not None:
            sock.close()
        
        print("Connection closed: %s:%d"%(self.address))
        
            
    def send(self, msg):
        """
        Args:
          -msg: str or bytes
        returns None if socket is currently not available for writing.
        """
        
        if type(msg) == str:
            msg = str(msg).encode('UTF-8')
        assert type(msg) is bytes
        
        lock = threading.Lock()
        lock.acquire()
        self.write_queue.append(msg)
        lock.release()
        
    
        
    def recv(self):
        """
        returns None if socket is currently not available for reading
        """
            
        lock = threading.Lock()
        lock.acquire()
        if len(self.read_queue) > 0:
            msg = self.read_queue.pop(0)
        else:
            msg = None
        lock.release()
        return msg
    
    
    def getSocket(self):
        assert False, "Must be implemented in derived class"
        

class StreamerClient(StreamerAbstract):
    def getSocket(self):
        return None
        
    def connectSocket(self, sock=None):
        return socket.create_connection(self.address, self.timeout)
        


class StreamerServer(StreamerAbstract):
    def getSocket(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(self.address)
        sock.listen(1)
        if self.timeout is not None:
            #sock.setblocking(False)
            sock.settimeout(self.timeout)
            print('Listener timeout has been set to %d msec'%int(self.timeout*1000))
        print("Server listening on address: %s:%d"%self.address)
        return sock
    
    def connectSocket(self, listener):
        sock, sock_addr = listener.accept()
        print('Connection established: %s:%d'%sock_addr)
        return sock
                
