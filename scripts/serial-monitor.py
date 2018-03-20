# -*- coding: utf-8 -*-
#!/usr/bin/python3

import serial
import binascii
from time import sleep
import sys
import datetime

port = '/dev/ttyUSB0'
baudrate = 9600
if len(sys.argv) > 2:
	port = sys.argv[1]
	baudrate = int(sys.argv[2])

channel_number = 4

with serial.Serial(port, baudrate, timeout=0.1) as ser:
    try:
        while True:
            raw_data = ser.read(256);
            if len(raw_data) != 0:
                if len(raw_data) == 5:
                    #hexCardID = binascii.hexlify(raw_data[4:0:-1]).decode()
                    hexCardID = binascii.hexlify(raw_data[1:]).decode()
                    channel = chr(raw_data[0])
                    if channel in [chr(i) for i in range(ord('A'),ord('A')+channel_number)]:
                        print("0x{} on channel {} at".format(hexCardID, channel, str(datetime.datetime.now())))
                else:
                    try:
                        print_text = raw_data.decode('ascii')
                    except Exception as e:
                        print_text = str(e) + ' at {}'.format(str(datetime.datetime.now())) + '\r\n'
                    finally:
                        print(print_text, end='')

    except KeyboardInterrupt as e:
        print("Bye.")


    
