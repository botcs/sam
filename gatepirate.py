#!/usr/bin/python3
# -*- coding: utf-8 -*-

import serial
from time import sleep, time, strftime
import sys
from datetime import datetime
import binascii

OKBLUE = '\033[94m'
GREEN = '\033[92m'
WARNING = '\033[93m'
RED = '\033[91m'
NO = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

class ITKGatePirate():
    def __init__(self, port='/dev/ttyUSB0', baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.channel_number = 1
        self.valid_channels = [chr(i) for i in range(ord('A'),ord('A')+self.channel_number)]
        self.sertimeout = 0.2
        self.reset_command = 'R'
        self.log_file_name = 'card_id_log.txt'

        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.sertimeout)
        except serial.serialutil.SerialException as e:
            print(RED+'Error! No serial device or port is locked!'+NO)
           

    def serial_read(self):
        raw_data = []
        try:
            raw_data = self.ser.read(256);
        except serial.serialutil.SerialException as e:
            print(RED+'Serial read error'+NO)

        return raw_data

    def serial_write(self, tx_data):
        byte_count = 0
        try:
            byte_count = self.ser.write(tx_data);
        except serial.serialutil.SerialException as e:
            print(RED+'Serial write error'+NO)

        return byte_count

    def time_now(self):
         t = int(time()*1000)
         return t

    def listen(self):
        from tendo import singleton
        try:
            me = singleton.SingleInstance( flavor_id="listener" ) # will sys.exit(-1) if other instance is running
        except singleton.SingleInstanceException as e:
            sys.exit(-1)
        print(OKBLUE+"CardID listener started at {}".format(strftime("%c"))+NO)
        try:
            with open(self.log_file_name, 'a') as logfile:
                while True:
                    raw_data = self.serial_read()
                    (cardid, channel, status) = self.process_raw_data(raw_data)
                    t_now = self.time_now()
                    if status == 0:
                        file_str = "{},{},{}\r\n".format(cardid, channel, t_now)
                        logfile.write(file_str)
                        logfile.flush()

                        print_str = "0x{} on channel {} at {}\r\n".format(cardid, channel,  str(datetime.fromtimestamp(int(t_now/1000))))
                        print(print_str, end='')
                    elif len(raw_data) != 0:
                        print("Status: {}, raw_data={}".format(status,raw_data))
        except KeyboardInterrupt as e:
            print(OKBLUE+BOLD+"Bye."+NO)
        finally:
            logfile.close()
            self.ser.close()

    def process_raw_data(self, raw_data):
        card_id_len = 1+4*2 # channel id + 32byte ascii cardID 
        if len(raw_data) == card_id_len: 
            channel = chr(raw_data[0])
            cardid = raw_data[1:]
            if channel in self.valid_channels:
                # TODO: wait for led access status
                status = 0
            else:
                status = 1
        elif len(raw_data) == 0:
            # no data was read
            cardid = None
            channel = None
            status = -1 
        else:
            # illegal data was read
            cardid = None
            channel = None
            status = -2
            
        if cardid is not None:
            cardid = cardid.decode('ascii')
        return (cardid, channel, status)
            
        
        
    # timeout in seconds
    def readOneCardID(self, timeout=None):
        if timeout is None:
            while True:
                raw_data = self.serial_read()
                t_now = self.time_now()
                (cardid, channel, status) = self.process_raw_data(raw_data)
                if status != -1:
                    break;
        else:
            # hope this works...
            self.ser.timeout = timeout
            raw_data = self.serial_read()

            # but this works... probably
#           tries = int((1/self.sertimeout)*timeout)
#           while(tries>0):
#               raw_data = self.ser.read(256);
#               if len(raw_data) == 1+4*2:
#                   break
#               tries -= 1

            # process raw_data
            t_now = self.time_now()
            (cardid, channel, status) = self.process_raw_data(raw_data)

            # reset ser.timeout to default
            self.ser.timeout = self.sertimeout

            return (cardid, t_now, channel, status)

    # send card id  to the server
    # accepts string of 8 hexadecimal digits, charachter of channel
    # returns access status
    # remark: only the channel 'A' is implemented in hardware yet. The controller will reject any other channel.
    #  (except 'R', which will trigger a soft reset.)
    def emulateCardID(self, cardid, channel='A'):
        if len(cardid) != 8:
            return -1
        if channel == self.reset_command:
            return -2
        if self.serial_write(channel.encode()) != 1:
            return -3
        if self.serial_write(bytes.fromhex(cardid)) != 4:
            return -4
        
        # TODO: wait for led access status
        status = 0
        return (status)
 
    def reset_mcu(self, wait_for_buffer_empty=True):
        if wait_for_buffer_empty:
            time.sleep(2) # recieving buffer must be empty
        if serial_write(self.reset_command.encode()) != 1:
            return -3

def main():
    if len(sys.argv) > 1:
        port = sys.argv[1]
        vau = ITKGatePirate(port=port)
    else:
        vau = ITKGatePirate()

    vau.listen()

if __name__ == "__main__":
    main()

