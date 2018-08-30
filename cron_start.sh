#!/bin/bash

cd /home/nvidia/sam/
SCREEN_NAME="mainscreen"
START_LOG="start_log.txt"
export DISPLAY=:0
xset -display ${DISPLAY} dpms force on

if ! screen -list | grep -q $SCREEN_NAME; then
  echo -n "Screen is NOT running, starting... " | tee -a $START_LOG
  #date | tee -a $START_LOG
  #screen -d -L -S video_split -m ffmpeg -s 640x480 -i /dev/video0 -codec copy -f v4l2 /dev/video1 -codec copy -f v4l2 /dev/video2
  #sleep 5
  screen -d -L -S $SCREEN_NAME -m python3 /home/nvidia/sam/client.py --display --cam 0 --fullscreen
else
  echo -n "Screen is running, not starting another. " | tee -a $START_LOG
  date | tee -a $START_LOG
fi

