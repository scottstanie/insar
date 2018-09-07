#!/bin/bash
set -e
SCRIPTNAME=`basename "$0"`

if [ "$#" -lt 1 ]
then
	echo 'Usage: $SCRIPTNAME "--args for --insar-process'
	echo 'Make sure to double quote arg string'
	exit 1
fi
COMMAND=$1
echo "Running command over each directory in current folder:"
FULL_CMD="insar process $COMMAND"
echo $FULL_CMD
echo "Will append --geojson with current directory's file"

for dirname in $(find -maxdepth 1 -type d ! -name "." | head -5); do
	echo "Moving to $dirname"
	geojson_file=$(eval "ls *.geojson | head -1")
	CUR_CMD="$FULL_CMD --geojson $geojson_file"
	echo $CUR_CMD
	eval $CUR_CMD
done
