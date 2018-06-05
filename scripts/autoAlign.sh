#!/bin/bash

SCREEN_NAME="autoAlign"
START_LOG="autoAlign_start_log.txt"
if ! screen -list | grep -q $SCREEN_NAME; then
  echo -n "Screen is NOT running, starting... " | tee -a $START_LOG
  date | tee -a $START_LOG
  screen -d -L -S $SCREEN_NAME -m numactl --cpunodebind=0 /home/csbotos/.virtualenvs/all/bin/python /home/csbotos/sam/scripts/autoAlign.py --workers 40 --root /home/csbotos/video/unprocessed-recordings/
else
  echo -n "Screen is running, not starting another. " | tee -a $START_LOG
  date | tee -a $START_LOG
fi

