#!/usr/bin/bash

google_drive_download() {
	ID=$1
	NAME=$2
	echo ""
	echo "---------------------------------------------------------------------------------------------------------------------------------------------"
	echo "downloading file ID $ID from google drive ... saving to $NAME"
	echo "---------------------------------------------------------------------------------------------------------------------------------------------"
	echo ""

	wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$ID -O- \
	     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt

	wget --load-cookies cookies.txt -O $NAME \
	     'https://docs.google.com/uc?export=download&id='$ID'&confirm='$(<confirm.txt)
	rm cookies.txt confirm.txt
     }

# Change this variable to download data to a different directory
DIR=data

# Exit script if directory already exists
if [ -d "$DIR" ]; then
	echo "Directory $DIR exists. Move this directory to download data from Google Drive"
	exit 1
fi

# Download the files from Jeff's Google Drive
mkdir $DIR
cd $DIR
compdata=predict-west-nile-virus.zip
moredata=chicago-west-nile-virus-mosquito-test-results.zip
weatherdata=NOAA-weather-data.csv

google_drive_download 1YrRTBPMeC3L2ilL71GGj37EAXkAbFT1d $compdata
google_drive_download 1BL-Rpz4AU7g14-6-jWc9sQt1Td3yvBQ3 $moredata
google_drive_download 1LpPStAaL6818teaGXCZWhGMSdwADhRjC $weatherdata

# Unzip files
echo ""
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo "Unzipping downloaded files"
echo "---------------------------------------------------------------------------------------------------------------------------------------------"
echo ""
unzip $compdata
unzip $moredata

for file in $PWD/*
do
	chmod 644 $file
done
