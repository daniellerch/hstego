#!/bin/bash

TIMEFORMAT="%R seconds"
CURR_FILE=`readlink -f $0`
DIR=`dirname $CURR_FILE`
cd $DIR

#echo "------------------------------------------------------------------------"
#echo "text + grayscale png"
#echo "------------------------------------------------------------------------"
#rm -f output-secret*
#echo -n "Embed: "
#time hstego.py embed input-secret.txt cover.png stego.png p4ssw0rd
#echo -n "Extract: "
#time hstego.py extract stego.png output-secret.txt p4ssw0rd
#if [ "`sha1sum input-secret.txt|cut -d' ' -f1`" != "`sha1sum output-secret.txt|cut -d' ' -f1`" ]
#then
#    echo "Extracting error (1)!";
#    diff input-secret.txt output-secret.txt
#    exit 0
#fi


#echo "------------------------------------------------------------------------"
#echo "bin + grayscale png"
#echo "------------------------------------------------------------------------"
#rm -f output-secret*
#echo -n "Embed: "
#time hstego.py embed input-secret.png cover.png stego.png p4ssw0rd
#echo -n "Extract: "
#time hstego.py extract stego.png output-secret.png p4ssw0rd
#if [ "`sha1sum input-secret.png|cut -d' ' -f1`" != "`sha1sum output-secret.png|cut -d' ' -f1`" ]
#then
#    echo "Extracting error (2)!";
#    exit 0
#fi


echo "------------------------------------------------------------------------"
echo "text + grayscale jpg"
echo "------------------------------------------------------------------------"
rm -f output-secret*
echo -n "Embed: "
time hstego.py embed input-secret.txt cover.jpg stego.jpg p4ssw0rd
echo -n "Extract: "
time hstego.py extract stego.jpg output-secret.txt p4ssw0rd
if [ "`sha1sum input-secret.txt|cut -d' ' -f1`" != "`sha1sum output-secret.txt|cut -d' ' -f1`" ]
then
    echo "Extracting error (3)!";
    diff input-secret.txt output-secret.txt
    exit 0
fi


#echo "------------------------------------------------------------------------"
#echo "text + color png"
#echo "------------------------------------------------------------------------"
#rm -f output-secret*
#echo -n "Embed: "
#time hstego.py embed input-secret.txt cover_color.png stego_color.png p4ssw0rd
#echo -n "Extract: "
#time hstego.py extract stego_color.png output-secret.txt p4ssw0rd
#if [ "`sha1sum input-secret.txt|cut -d' ' -f1`" != "`sha1sum output-secret.txt|cut -d' ' -f1`" ]
#then
#    echo "Extracting error (4)!";
#    diff input-secret.txt output-secret.txt
#    exit 0
#fi


#echo "------------------------------------------------------------------------"
#echo "text + color jpg"
#echo "------------------------------------------------------------------------"
#rm -f output-secret*
#echo -n "Embed: "
#time hstego.py embed input-secret.txt cover_color.jpg stego_color.jpg p4ssw0rd
#echo -n "Extract: "
#time hstego.py extract stego_color.jpg output-secret.txt p4ssw0rd
#if [ "`sha1sum input-secret.txt|cut -d' ' -f1`" != "`sha1sum output-secret.txt|cut -d' ' -f1`" ]
#then
#    echo "Extracting error (5)!";
#    diff input-secret.txt output-secret.txt
#    exit 0
#fi





