#!/bin/bash
# TODO: pass in the path: currently only runs on the current directory
# Wrapper to run in parallel:
call_snaphu() {
	INTFILE=$1
	CORNAME=$(echo $INTFILE | sed 's/.int/.cc/' )
	OUTNAME=$(echo $INTFILE | sed 's/.int/.unw/' )
	echo "Running snaphu on $INTFILE: output to $OUTNAME"
	# ~/phase_upwrap/bin/snaphu -s $INTFILE $LINELENGTH -c $CORNAME -o $OUTNAME;
	return 0;
}
# Need to export so that subprocesses called by xargs have call_snaphu
export -f call_snaphu


SCRIPTNAME=`basename "$0"`
# aka width aka xsize aka number of samples in one range line
LINELENGTH=$1
if [ -z $LINELENGTH ]; then
	echo "Usage: $SCRIPTNAME LINELENGTH"
	exit 1
fi


find . -name "*.int" | xargs --max-procs=20 -I {} $SHELL -c 'call_snaphu {} $LINELENGTH'

