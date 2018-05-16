#!/usr/bin/python3
# -*- coding: utf-8 -*-

import gatepirate
from bcolors import *
from time import sleep, time
from datetime import datetime

do_forever = True
while do_forever:
    try:
        vau = gatepirate.ITKGatePirate(pmode='serial')
        vau.listen()
    except KeyboardInterrupt as e:
        print(OKBLUE+BOLD+"Bye!"+NO)
        do_forever = False
    except Exception as e:
        msg_str="Exception occured at {}, restarting soon...".format(str(datetime.fromtimestamp(int(time()))))
        print(RED+msg_str)
        print(str(e))
        print(NO)
        with open('card_log.error.log', 'a') as logfile:
            logfile.write(msg_str+"\n")
            logfile.write(str(e)+"\n")
        sleep(15)


