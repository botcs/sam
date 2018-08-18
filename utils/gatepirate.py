#!/usr/bin/python3
# -*- coding: utf-8 -*-

import serial
from time import sleep, time, strftime
import sys
from datetime import datetime
import binascii
import glob

import os

#sys.path.insert(0, '/home/nvidia/sam/utils')
sys.path.insert(0, '/home/botoscs/tegra-home/sam/utils')
from sqlrequest import db_query, initDB

base_dir = '/home/nvidia/card_log/'
#base_dir = '/home/levi/kibu/card/'
#base_dir = '/tmp/'

OKBLUE = '\033[94m'
GREEN = '\033[92m'
WARNING = '\033[93m'
RED = '\033[91m'
NO = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

class ITKGatePirate():
    def __init__(self, port='auto', baudrate=9600, pmode='daemon'):
        
        initDB()    
        if port == 'auto':
            self.port = glob.glob('/dev/ttyUSB*')[0]
            #self.port = glob.glob('/dev/ttyACM*')[0]
        else:
            self.port = port

        self.baudrate = baudrate
        self.channel_number = 1
        self.valid_channels = [chr(i) for i in range(ord('A'),ord('A')+self.channel_number)]
        self.sertimeout = 0
        self.reset_command = 'R'
        self.get_led_command = 'L'
        self.log_file_name = 'card_id_log.txt'
        self._mode = pmode

        if self._mode == 'serial':
            try:
                self.ser = serial.Serial(self.port, self.baudrate, timeout=self.sertimeout)
            except serial.serialutil.SerialException as e:
                print(RED+'Error! No serial device or port is locked!'+NO)
        elif self._mode == 'daemon':
            #TODO 
            pass
        else:
            print('invalid mode')
            
        # Make sure files exists, and are empty
        open(os.path.join(base_dir, 'emulate.txt'), 'w')
        open(os.path.join(base_dir, 'last.txt'), 'w')

        self.last_raw_data = ""

    def serial_read(self, num=64, timeout=300):
        raw_data = bytearray(b'')
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

    def serial_write(self, tx_data):
        byte_count = 0
        try:
            byte_count = self.ser.write(tx_data);
        except serial.serialutil.SerialException as e:
            print(RED+'Serial write error'+NO)

        return byte_count

    def time_now(self):
        # Returns miliseconds in linux epoch time
        t = int(time()*1000)
        return t
         
    def SQLInsert(self, cardid, channel, t_now, status, emulated):
        emulated = int(emulated)
        SQL_INSERT = """
            INSERT INTO card_log(card_ID, timestamp, gate, emulated)
            VALUES("{card_ID}", {timestamp}, "{gate}", {emulated})
        """.format(card_ID=cardid, timestamp=t_now, gate=channel, emulated=emulated)
        db_query(SQL_INSERT)


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
                
                    ### READ
                    raw_data = self.serial_read(num=9)
                    (cardid, channel, status) = self.process_raw_data(raw_data)
                    t_now = self.time_now()
                    if status == 0 or status == 1:
                        file_str = "{},{},{},{}\r\n".format(cardid, channel, t_now, status)
                        logfile.write(file_str)
                        logfile.flush()
                        try:
                            self.SQLInsert(
                                cardid=cardid, 
                                channel=channel, 
                                t_now=t_now, 
                                status=status, 
                                emulated=False)
                        except (Exception) as e:
                            print(e)

                        with open(os.path.join(base_dir, 'last.txt'), 'w') as out:
                            out.write(file_str)
                        status_str = "OK" if status==0 else "SHALL NOT PASS"
                        print_str = "0x{id} {status} on channel {ch} at {time}\r\n".format(
                            id=cardid, 
                            ch=channel, 
                            status=status_str, 
                            time=str(datetime.fromtimestamp(int(t_now/1000))))
                        print(print_str, end='')
                    elif len(raw_data) != 0:
                        print("Status: {}, raw_data={}".format(status,raw_data))
                        
                        
                    ### EMULATE
                    try:
                        emf = open(os.path.join(base_dir, 'emulate.txt'), 'r')
                        a = emf.read().splitlines()
                        if len(a) > 0:
                            if len(a[0]) == 8:
                                cardid = a[0]
                                # HARDCODED FOR A SINGLE ENTRY
                                # TODO: Multiple entry
                                channel = 'A'
                                status = self._emulateCardID(cardid=cardid, channel=channel)
                                try:
                                    self.SQLInsert(
                                        cardid=cardid, 
                                        channel=channel, 
                                        t_now=t_now, 
                                        status=status, 
                                        emulated=True)
                                except (Exception) as e:
                                    print(e)
                                self._force_open()
                            else:
                                print("Invalid data read from emulate.txt (<{data}>)".format(data=a[0]))
                            emf.close()
                            with open(os.path.join(base_dir, 'emulate.txt'), 'w') as clear_file:
                                pass
                    except IOError:
                        print("Could not read emulate.txt")
        finally:
            logfile.close()
            self.ser.close()

    def process_raw_cardid_input(self, cardid_str):
        channel = chr(cardid_str[0])
        cardid = cardid_str[1:9]
        #status_chr = chr(cardid_str[9])
        #if channel in self.valid_channels:
        #    status = self.process_entry_status(cardid_str[9:10])
        #else:
        #    status = None

        return (cardid, channel, 0)

    def process_raw_data(self, raw_data):
        card_id_len = 1+4*2 # channel id + 32byte ascii cardID + 1 char status
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
            
        if channel == 'W' or channel == 'i':
            cardid = None
            channel = None
            status = -2
            
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

    # timeout in seconds
    def _readOneCardID(self, timeout=None):
        if timeout is None:
            while True:
                raw_data = self.serial_read()
                t_now = self.time_now()
                (cardid, channel, status) = self.process_raw_data(raw_data)
                if status != -1:
                    break;
        else:
            self.ser.timeout = timeout
            raw_data = self.serial_read()

            # process raw_data
            t_now = self.time_now()
            (cardid, channel, status) = self.process_raw_data(raw_data)

            # reset ser.timeout to default
            self.ser.timeout = self.sertimeout

            return (cardid, t_now, channel, status)

    def emulateCardID(self, cardid, channel='A'):
        t_now = self.time_now()
        with open(os.path.join(base_dir, 'emulate.txt'), 'w') as emuf:
            emuf.write(cardid)

        
    def readCardID(self, max_age=None):
        with open(os.path.join(base_dir, 'last.txt'),'r') as last_card:
            x = last_card.read().split(',')
        with open(os.path.join(base_dir, 'last.txt'),'w') as last_card:
            pass
        if max_age is not None and 2 < len(x):
            if (self.time_now() - int(x[2])) > max_age:
                return []
                
        return x

    # send card id  to the server
    # accepts string of 8 hexadecimal digits, charachter of channel
    # returns access status
    # remark: only the channel 'A' is implemented in hardware yet. The controller will reject any other channel.
    #  (except 'R', which will trigger a soft reset.)
    def _emulateCardID(self, cardid, channel='A'):
        self.empty_serial_buffer()

        if len(cardid) != 8:
            return -1
        if channel == self.reset_command:
            return -2
        if self.serial_write(channel.encode()) != 1:
            return -3
        if self.serial_write(bytes.fromhex(cardid)) != 4:
            return -4

        # wait for led access status
        # status = self.read_entry_status()
        return 0


    def reset_mcu(self, wait_for_buffer_empty=True):
        if wait_for_buffer_empty:
            sleep(2) # recieving buffer must be empty
        if self.serial_write(self.reset_command.encode()) != 1:
            return -3

        sleep(0.1)
        self.empty_serial_buffer()


    def empty_serial_buffer(self):
        to = self.ser.timeout
        self.ser.timeout = 0
        while True:
            buf = self.ser.read(64)
            if len(buf) != 0:
               print(buf)
            else:
               break
        self.ser.timeout = to
        


    def _force_open(self):
        if self.serial_write(b'O') != 1:
            return -3
        sleep(0.1)
        self.empty_serial_buffer()
        return 0

    def _clear_status(self):
        if self.serial_write(b'C') != 1:
            return -3
        sleep(0.1)
        self.empty_serial_buffer()
        return 0



