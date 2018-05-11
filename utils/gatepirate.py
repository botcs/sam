#!/usr/bin/python3
# -*- coding: utf-8 -*-

import serial
from time import sleep, time, strftime
import sys
from datetime import datetime
import binascii
import glob

sys.path.insert(0, '/home/nvidia/sam/')
from utils import send_query


OKBLUE = '\033[94m'
GREEN = '\033[92m'
WARNING = '\033[93m'
RED = '\033[91m'
NO = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

class ITKGatePirate():
    def __init__(self, port='auto', baudrate=9600):
        if port == 'auto':
            self.port = glob.glob('/dev/ttyUSB*')[0]
        else:
            self.port = port
        self.baudrate = baudrate
        self.channel_number = 1
        self.valid_channels = [chr(i) for i in range(ord('A'),ord('A')+self.channel_number)]
        self.sertimeout = 0
        self.reset_command = 'R'
        self.log_file_name = 'card_id_log.txt'

        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.sertimeout)
        except serial.serialutil.SerialException as e:
            print(RED+'Error! No serial device or port is locked!'+NO)

        self.last_raw_data = ""

    def serial_read(self, num=64, timeout=300):
        raw_data = bytearray(b'')
        #timeout = 300 # ms
        to = self.ser.timeout
        self.ser.timeout = 0
        try:
            tm = int(time()*1000)
            while len(raw_data) < num and (int(time()*1000)-tm < timeout):
                sleep(0.01)
                ch = self.ser.read(1);
                if len(ch) != 0:
                    tm = int(time()*1000)
                    raw_data.append(ord(ch))

        except serial.serialutil.SerialException as e:
            print(RED+'Serial read error'+NO)
        self.ser.timeout = to
        return raw_data

    def serial_read_old(self):
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

    def SQLInsert(self, cardid, channel, t_now, status):
        SQL_INSERT = """
            INSERT INTO card_log(card_ID, timestamp, gate, success)
            VALUES("{card_ID}", {timestamp}, "{gate}", {success})
        """.format(card_ID=cardid, timestamp=t_now, gate=channel, success=status)
        send_query(SQL_INSERT)


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
                    if status == 0 or status == 1:
                        file_str = "{},{},{},{}\r\n".format(cardid, channel, t_now, status)
                        logfile.write(file_str)
                        logfile.flush()
                        try:
                            self.SQLInsert(cardid=cardid, channel=channel, t_now=t_now, status=status)
                        except RuntimeError as e:
                            print(e)
                        status_str = "OK" if status==0 else "SHALL NOT PASS"
                        print_str = "0x{id} {status} on channel {ch} at {tim}\r\n".format(id=cardid, ch=channel, status=status_str, tim=str(datetime.fromtimestamp(int(t_now/1000))))
                        print(print_str, end='')
                    elif len(raw_data) != 0:
                        print("Status: {}, raw_data={}".format(status,raw_data))
        finally:
            logfile.close()
            self.ser.close()

    def process_raw_cardid_input(self, cardid_str):
        channel = chr(cardid_str[0])
        cardid = cardid_str[1:9]
        #status_chr = chr(cardid_str[9])
        if channel in self.valid_channels:
            status = self.process_entry_status(cardid_str[9:10])
        else:
            status = None

        return (cardid, channel, status)

    def process_raw_data(self, raw_data):
        card_id_len = 1+4*2+1 # channel id + 32byte ascii cardID + 1 char status
        if len(raw_data) == card_id_len:
            (cardid, channel, status) = self.process_raw_cardid_input(raw_data)
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
            if len(self.last_raw_data) > 0:
                reconstructed_data = self.last_raw_data + raw_data
                if chr(reconstructed_data[0]) in self.valid_channels and len(reconstructed_data) == card_id_len:
                    print(WARNING+"kette tores eszlelve"+NO)
                    # ketté tört az adat
                    (cardid, channel, status) = self.process_raw_cardid_input(reconstructed_data)

        self.last_raw_data = raw_data
        if cardid is not None:
            cardid = cardid.decode('ascii')
        return (cardid, channel, status)


    def process_entry_status(self, raw_data):
        if len(raw_data) == 0:
            #raise noAnswerException
            status = None
        elif len(raw_data) == 1:
            if raw_data[0] == ord('+'):
                status = 0
            elif raw_data[0] == ord('-'):
                status = 1
            else:
                status = None
        else:
            print("err")
            #raise serialMisalignedException
            status = None

        return status

    def read_entry_status(self):
        raw_data = self.serial_read(num=1, timeout=1000)
        status = self.process_entry_status(raw_data)
        return status


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

        # wait for led access status
        status = self.read_entry_status()
        return (status)


    def reset_mcu(self, wait_for_buffer_empty=True):
        if wait_for_buffer_empty:
            time.sleep(2) # recieving buffer must be empty
        if serial_write(self.reset_command.encode()) != 1:
            return -3

