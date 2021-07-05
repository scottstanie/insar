#!/bin/bash
# Quick helper to count the number of unique dates of all igrams created
# Also prints the size of the slclist for comparison to see if any 
# dates are missing from igrams

set -e

if [ "$#" -lt 1 ]
then
	DIR="."
else
	DIR="$1"
fi


echo "Searching in $DIR for ifglist"
INTFILE="$DIR/ifglist"
if [ ! -f "$INTFILE" ]; then
	echo "No ifglist file."
	exit 1
fi

# Grab the date on each side of the igram name
cut -d'_' -f1 "$INTFILE" | sort | uniq > "$DIR/start_dates.txt"
cut -d'_' -f2 "$INTFILE" | cut -d'.' -f1 | sort | uniq > "$DIR/end_dates.txt"

echo "Writing unique igram dates to $DIR/int_dates"
cat $DIR/start_dates.txt $DIR/end_dates.txt | sort | uniq > "$DIR/int_dates.txt"

echo "Total number igrams in ifglist:"
wc -l $DIR/ifglist

echo "Number of unique dates in the ifglist:"
wc -l $DIR/int_dates.txt

# Trim the .geo file name to just the date
echo "Writing unique geo dates to $DIR/geo_dates"
cat "$DIR/slclist" | cut -d'_' -f 6 | cut -d'T' -f1 | sort | uniq > "$DIR/geo_dates.txt"
echo "Number of unique dates in the slclist:"
wc -l $DIR/geo_dates.txt


