# -*- coding: utf-8 -*-
#!/usr/bin/python3

import serial
import binascii
from time import sleep
import sys
import datetime
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True, help='Path to buffer directory')
parser.add_argument('--port', default='/dev/ttyUSB0')
parser.add_argument('--baudrate', type=int, default=9600)
args = parser.parse_args()


channel_number = 4


if not os.path.exists(args.path):
	raise RuntimeError('Cannot find %40s' % args.path)



def TimeStamp():
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d.%H-%M-%S.%f')

with serial.Serial(args.port, args.baudrate, timeout=0.1) as ser:
    try:
        while True:
            raw_data = ser.read(256);
            if len(raw_data) != 0:
                if len(raw_data) == 5:
                    #hexCardID = binascii.hexlify(raw_data[4:0:-1]).decode()
                    hexCardID = binascii.hexlify(raw_data[1:]).decode()
                    channel = chr(raw_data[0])
                    if channel in [chr(i) for i in range(ord('A'),ord('A')+channel_number)]:
                        print("0x{} on channel {} at".format(hexCardID, channel, TimeStamp()))
                        glob.glob(args.path + '/', recursive=True)


                else:
                    try:
                        print_text = raw_data.decode('ascii')
                    except Exception as e:
                        print_text = str(e) + ' at {}'.format(str(datetime.datetime.now())) + '\r\n'
                    finally:
                        print(print_text, end='')

    except KeyboardInterrupt as e:
        print("Bye.")
