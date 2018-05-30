#!/usr/bin/bash

for INTFILE in `ls -1 *.int`; do 
	CORNAME=$(echo $INTFILE | sed 's/.int/.cc/' ) 
	OUTNAME=$(echo $INTFILE | sed 's/.int/.unw/' ) 
	echo "Running snaphu on $INTFILE: output to $OUTNAME"
	~/phase_upwrap/bin/snaphu -s $INTFILE 1000 -c $CORNAME -o $OUTNAME; 
done

