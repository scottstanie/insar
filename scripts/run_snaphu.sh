#!/usr/bin/bash

SCRIPTNAME=`basename "$0"`
# aka width aka xsize aka number of samples in one range line
LINELENGTH=$1
if [ -z $LINELENGTH ]; then
	echo "Usage: $SCRIPTNAME LINELENGTH"
	exit 1
fi


for INTFILE in `ls -1 *.int`; do
	CORNAME=$(echo $INTFILE | sed 's/.int/.cc/' )
	OUTNAME=$(echo $INTFILE | sed 's/.int/.unw/' )
	echo "Running snaphu on $INTFILE: output to $OUTNAME"
	~/phase_upwrap/bin/snaphu -s $INTFILE $LINELENGTH -c $CORNAME -o $OUTNAME;
done

