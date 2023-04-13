#!/bin/bash
while [ "1" = "1" ]
do
 CUDA_VISIBLE_DEVICES=2,3,4,5 python3 train.py
done
