#!/bin/bash

ssh 192.168.1.2 python ~/study/00_launch.py 1 &
ssh 192.168.1.3 python ~/study/00_launch.py 2 &
ssh 192.168.1.4 python ~/study/00_launch.py 3 &
python ~/study/00_launch.py 0 