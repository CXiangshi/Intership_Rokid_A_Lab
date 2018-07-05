#!/bin/bash

$path = /home/ccc/data/youtube_noise
 

function download(){
	for line in $(cat url.txt)
	do 
		youtube-dl  -x --audio-format "wav" -o "$a.%(ext)s" $line
		#let a=a+1	
	done
	if $(ls -l|grep "^-"|wc -l)<$a
		then 
			download $a
	else
		echo "done!"
}
a=1
for file in $(ls $path)
do
	cd ${path}/$file
	for line in $(cat url.txt)
	do 
		youtube-dl  -x --audio-format "wav" -o "$a.%(ext)s" $line
		let a=a+1	
	done
done
