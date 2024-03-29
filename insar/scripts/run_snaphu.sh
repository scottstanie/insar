#!/bin/bash
set -e
# TODO: pass in the path: currently only runs on the current directory
# Wrapper to run in parallel:
if [ -z $PHASE_UNWRAP_DIR ]; then
  PHASE_UNWRAP_DIR=~/phase_unwrap/bin
fi
echo "Directory to find snaphu:"
echo "$PHASE_UNWRAP_DIR"

call_snaphu() {
	INTFILE=$1
	WIDTH=$2
	CORNAME=$(echo $INTFILE | sed 's/.int/.cc/' | sed 's/.lowpass//' )
	OUTNAME=$(echo $INTFILE | sed 's/.int/.unw/' | sed 's/.lowpass//' )
	echo "Running snaphu on $INTFILE with width $WIDTH: output to $OUTNAME"
    if [ -f $OUTNAME ]; then
        echo "$OUTNAME exists already. Skipping unwrapping $INTFILE"
    else
        if [ $WIDTH -gt 1000 ]; then
            echo "$PHASE_UNWRAP_DIR/snaphu -s $INTFILE $WIDTH -c $CORNAME -o $OUTNAME -S --tile 3 3 30 30 --nproc 9"
            $PHASE_UNWRAP_DIR/snaphu -s $INTFILE $WIDTH -c $CORNAME -o $OUTNAME -S --tile 3 3 30 30 --nproc 9;
        # elif [ $WIDTH -gt 500 ]; then
        #     $PHASE_UNWRAP_DIR/snaphu -s $INTFILE $WIDTH -c $CORNAME -o $OUTNAME -S --tile 2 2 30 30 --nproc 4;
        else
            echo "$PHASE_UNWRAP_DIR/snaphu -s $INTFILE $WIDTH -c $CORNAME -o $OUTNAME" ;
            $PHASE_UNWRAP_DIR/snaphu -s $INTFILE $WIDTH -c $CORNAME -o $OUTNAME ;
        fi
        echo "Finished unwrapping $INTFILE "
    fi
}

SCRIPTNAME=`basename "$0"`
if [ "$#" -lt 1 ]
then
	echo "Usage: $SCRIPTNAME width [lowpass-box-size=1] [max-jobs=num-cores]"
	exit 1
fi
# aka linelength aka xsize aka number of samples in one range line
WIDTH=$1

# If they didn't pass in third arg for numver of jobs:
if [ "$#" -lt 3 ]
then
	# Get the number of cores (at least on linux): if it fails, use 4
	# NUM_CORES=$(grep -c ^processor /proc/cpuinfo) && MAX_PROCS=$NUM_CORES || MAX_PROCS=4
    MAX_PROCS=10
else
	MAX_PROCS=$3
fi
echo "Running with number of jobs=$MAX_PROCS "


# If boxsize not passed, or boxsize < 2
if [ "$#" -lt 2 ] || [ $2 -lt 2 ]
then
	echo "Skipping low pass filter"
	SNAFU_FILE_EXT=".int"
else
	BOX_SIZE=$2
 	LOWPASS=$PHASE_UNWRAP_DIR/lowpass
	echo "Running $LOWPASS with box size $BOX_SIZE"
    

	# For loop is faster for the fortran program than xargs
	for FILE in $(find . -maxdepth 1 -name "*.int"); do
		$LOWPASS $FILE $WIDTH $BOX_SIZE
	done
	SNAFU_FILE_EXT=".int.lowpass"
fi

# Need to export so that subprocesses called by xargs have call_snaphu and vars
export -f call_snaphu
export PHASE_UNWRAP_DIR
export WIDTH
export MAX_PROCS

# Call snaphu with each file name matched by SNAFU_FILE_EXT, and pass WIDTH to each call
find . -maxdepth 1 -name "*${SNAFU_FILE_EXT}" -print0 | xargs -0 --max-procs=$MAX_PROCS -I{} $SHELL -c "call_snaphu {} $WIDTH"

