#!/bin/bash

# Removes references to a bad .geo date within the ifglist and slclist
# Used when there is any reason to completely remove all igrams
# for one specific date
# If no path specified, igrams assumed to be in current directory
# Date format should be YYYYmmdd



if [ "$#" -lt 1 ]
then
	echo "Usage: $0 YYYYmmdd [path]"
	exit 1
else
	RMDATE="$1"
fi

if [ "$#" -lt 2 ]
then
	DIR="."
else
	DIR="$2"
fi

if [ ! -f "$DIR/ifglist" ]; then
	echo "File $DIR/ifglist does not exist."
	exit 1
fi


NUMLINES=$(grep -c "$RMDATE" "$DIR/ifglist")
if [ $NUMLINES -gt 0 ]; then
	echo "Removing $NUMLINES lines from $DIR/ifglist that match $RMDATE"
	grep -v "$RMDATE" "$DIR/ifglist" > /tmp/intremove && mv /tmp/intremove "$DIR/ifglist"
else
	echo "No lines matching $RMDATE in the $DIR/ifglist."
fi


NUMLINES=$(grep -c "$RMDATE" "$DIR/slclist")
if [ $NUMLINES -gt 0 ]; then
	echo "Removing $NUMLINES lines from $DIR/slclist that match $RMDATE"
	grep -v "$RMDATE" "$DIR/slclist" > /tmp/intremove && mv /tmp/intremove "$DIR/slclist"
else
	echo "No lines matching $RMDATE in the $DIR/slclist."
fi
