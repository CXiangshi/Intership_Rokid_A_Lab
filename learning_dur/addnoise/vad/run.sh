#!/bin/bash
make

export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
path=/home/xdwang/wavNOISE/awake_add_noise

for x in rokid_online_p1 rokid_online_p2 rokid_online_p3; do
	./vadtest $path/${x}_wavlist.text $path/${x}_vad.text
	#cat result.txt
done
#rm result.txt
#make clean
