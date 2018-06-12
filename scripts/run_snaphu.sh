#!/usr/bin/bash

SCRIPTNAME=`basename "$0"`
# aka length aka xsize aka number of samples in one range line
LENGTH=$1
if [ -z $LENGTH ]; then
	echo "Usage: $SCRIPTNAME LENGTH"
	exit 1
fi


for INTFILE in `ls -1 *.int`; do
	CORNAME=$(echo $INTFILE | sed 's/.int/.cc/' )
	OUTNAME=$(echo $INTFILE | sed 's/.int/.unw/' )
	echo "Running snaphu on $INTFILE: output to $OUTNAME"
	~/phase_upwrap/bin/snaphu -s $INTFILE $LENGTH -c $CORNAME -o $OUTNAME;
done


for i in *.int ; do dismphfile $i 800 ; mv dismph.tif `echo $i | sed 's/int$/tif/'` ; done
