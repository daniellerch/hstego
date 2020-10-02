#!/bin/bash


CURR_FILE=`readlink -f $0`
DIR=`dirname $CURR_FILE`
cd $DIR

rm -f output-secret*
hstego.py embed input-secret.txt cover.png stego.png p4ssw0rd
hstego.py extract stego.png output-secret.txt p4ssw0rd
if [ "`sha1sum input-secret.txt|cut -d' ' -f1`" != "`sha1sum output-secret.txt|cut -d' ' -f1`" ]
then
    echo "Extracting error (1)!";
    diff input-secret.txt output-secret.txt
    exit 0
fi


rm -f output-secret*
hstego.py embed input-secret.png cover.png stego.png p4ssw0rd
hstego.py extract stego.png output-secret.png p4ssw0rd
if [ "`sha1sum input-secret.png|cut -d' ' -f1`" != "`sha1sum output-secret.png|cut -d' ' -f1`" ]
then
    echo "Extracting error (2)!";
    exit 0
fi


rm -f output-secret*
hstego.py embed input-secret.txt cover.jpg stego.jpg p4ssw0rd
hstego.py extract stego.jpg output-secret.txt p4ssw0rd
if [ "`sha1sum input-secret.txt|cut -d' ' -f1`" != "`sha1sum output-secret.txt|cut -d' ' -f1`" ]
then
    echo "Extracting error (3)!";
    diff input-secret.txt output-secret.txt
    exit 0
fi





