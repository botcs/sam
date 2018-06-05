#!/bin/bash

SCREEN_NAME="autoSQL"
START_LOG="autoSQL_start_log.txt"
if ! screen -list | grep -q $SCREEN_NAME; then
  echo -n "Screen is NOT running, starting... " | tee -a $START_LOG
  date | tee -a $START_LOG
  screen -d -L -S $SCREEN_NAME -m /home/csbotos/.virtualenvs/all/bin/python3 /home/csbotos/sam/scripts/autoSQL.py
else
  echo -n "Screen is running, not starting another. " | tee -a $START_LOG
  date | tee -a $START_LOG
fi


