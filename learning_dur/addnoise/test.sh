#!/bin/bash

a=1
for line in $(cat test_url.txt)
do 
	youtube-dl -o $a $line
	let a=a+1
done
