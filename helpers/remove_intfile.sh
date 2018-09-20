#!/bin/bash

# Removes references to a bad .geo date within the intlist and geolist
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

if [ ! -f "$DIR/intlist" ]; then
	echo "File $DIR/intlist does not exist."
	exit 1
fi


NUMLINES=$(grep -c "$RMDATE" "$DIR/intlist")
if [ $NUMLINES -gt 0 ]; then
	echo "Removing $NUMLINES lines from $DIR/intlist that match $RMDATE"
	grep -v "$RMDATE" "$DIR/intlist" > /tmp/intremove && mv /tmp/intremove "$DIR/intlist"
else
	echo "No lines matching $RMDATE in the $DIR/intlist."
fi


NUMLINES=$(grep -c "$RMDATE" "$DIR/geolist")
if [ $NUMLINES -gt 0 ]; then
	echo "Removing $NUMLINES lines from $DIR/geolist that match $RMDATE"
	grep -v "$RMDATE" "$DIR/geolist" > /tmp/intremove && mv /tmp/intremove "$DIR/geolist"
else
	echo "No lines matching $RMDATE in the $DIR/geolist."
fi
