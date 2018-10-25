#!/bin/bash
set -e
SCRIPTNAME=`basename "$0"`

if [ "$#" -lt 1 ]
then
	echo 'Usage: $SCRIPTNAME "--args for insar process" [DIRLIST.txt] [--force-run]'
	echo "Runs over pre-unzipped directories"
	echo "Make sure to double quote arg string"
	echo "DIRLIST.txt is optional file contining the names of directories"
	echo "--force-run is 1 or 0 (default 0, will skip a directory if no .SAFE files inside)"
	exit 1
fi
COMMAND=$1

if [ "$#" -gt 1 ]
then
	echo "Using $2 for directory list"
	DIRLIST=$(cat $2)
else
	# Order the directories by number of .SAFE files contained in them
	# for dirname in $(find -maxdepth 1 -type d ! -name "."); do
	echo 'Sorting directories by number of .SAFE directories'
	find . -name "S1[AB]*.SAFE" |  cut -d'/' -f2 | sort | uniq -c | sort -nr | awk '{print $2}' > directory_order.txt
	DIRLIST=$(cat directory_order.txt)
fi

if [ "$#" -gt 2 ]
then
	FORCE=$3
else
	FORCE=0
fi

echo "Running command over each directory in current folder:"
FULL_CMD="insar process --no-unzip $COMMAND "
echo $FULL_CMD
echo "Will append --geojson with current directory's file"

for dirname in $DIRLIST; do
	# Skip if no sentinel files to run on
	SENT_FILES=$(find $dirname -maxdepth 1 -name "S1[AB]*.SAFE" )
	if [ "$FORCE" -ne 1 ]; then
		if [ -z "$SENT_FILES" ]; then
			echo "Skipping $dirname, no sentinel files"
			continue
		fi
	fi

	echo "Moving to $dirname"
	cd "$dirname"

	geojson_file=$(eval "ls *.geojson | head -1")
	if [ -z "$geojson_file" ]; then  # If it is zero length:
		CUR_CMD="$FULL_CMD"
	else
		CUR_CMD="$FULL_CMD --geojson $geojson_file"
	fi
	echo $CUR_CMD
	eval $CUR_CMD

	echo "Done with $dirname, moving up 1 dir"
	cd ..
done
