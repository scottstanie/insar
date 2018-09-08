#!/bin/bash
set -e
SCRIPTNAME=`basename "$0"`

if [ "$#" -lt 1 ]
then
	echo 'Usage: $SCRIPTNAME "--args for --insar-process"'
	echo "Runs over pre-unzipped directories"
	echo "Make sure to double quote arg string"
	exit 1
fi
COMMAND=$1
echo "Running command over each directory in current folder:"
FULL_CMD="insar process --no-unzip $COMMAND "
echo $FULL_CMD
echo "Will append --geojson with current directory's file"

for dirname in $(find -maxdepth 1 -type d ! -name "." | head -5); do
	# Skip if no sentinel files to run on
	SENT_FILES=$(find $dirname -name "S1[AB]*.SAFE" )
	if [ -z $SENT_FILES ]; then
		echo "Skipping $dirname, no sentinel files"
		continue
	fi

	echo "Moving to $dirname"
	cd "$dirname"

	geojson_file=$(eval "ls *.geojson | head -1")
	CUR_CMD="$FULL_CMD --geojson $geojson_file"
	echo $CUR_CMD
	eval $CUR_CMD

	echo "Done with $dirname, moving up 1 dir"
	cd ..
done
