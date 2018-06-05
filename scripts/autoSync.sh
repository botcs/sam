#!/bin/bash

SCREEN_NAME="autoSync"
START_LOG="autoSync_start_log.txt"
if ! screen -list | grep -q $SCREEN_NAME; then
  echo -n "Screen is NOT running, starting... " | tee -a $START_LOG
  date | tee -a $START_LOG
  screen -d -L -S $SCREEN_NAME -m rsync --remove-source-files -hvre ssh tegra:/home/nvidia/sam/recordings /home/csbotos/video/unprocessed-recordings/ --chmod=Du=rwx,Dg=rx,Do=x,Fu=rw,Fg=rx,Fo=x
else
  echo -n "Screen is running, not starting another. " | tee -a $START_LOG
  date | tee -a $START_LOG
fi

