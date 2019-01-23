#!/bin/bash

#Enable Fan Control
nvidia-settings -a '[gpu:0]/GPUFanControlState=1'

#Overclock Memory
nvidia-settings -c :0 -a '[gpu:0]/GPUMemoryTransferRateOffset[3]=1000'

#Overclock GPU
nvidia-settings -c :0 -a '[gpu:0]/GPUGraphicsClockOffset[3]=350'