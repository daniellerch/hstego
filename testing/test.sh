#!/bin/bash


echo "------------------------------------------------------------------------"
echo "text + grayscale png"
echo "------------------------------------------------------------------------"
rm -f testing/output-secret*
echo -n "Embed: "
time ./hstego.py embed testing/input-secret.txt testing/cover.png testing/stego.png p4ssw0rd
echo -n "Extract: "
time ./hstego.py extract testing/stego.png testing/output-secret.txt p4ssw0rd
if [ "`sha1sum testing/input-secret.txt|cut -d' ' -f1`" != "`sha1sum testing/output-secret.txt|cut -d' ' -f1`" ]
then
    echo "Extracting error (1)!";
    diff testing/input-secret.txt testing/output-secret.txt
    exit 0
fi




echo "------------------------------------------------------------------------"
echo "bin + grayscale png"
echo "------------------------------------------------------------------------"
rm -f testing/output-secret*
echo -n "Embed: "
time ./hstego.py embed testing/input-secret.png testing/cover.png testing/stego.png p4ssw0rd
echo -n "Extract: "
time ./hstego.py extract testing/stego.png testing/output-secret.png p4ssw0rd
if [ "`sha1sum testing/input-secret.png|cut -d' ' -f1`" != "`sha1sum testing/output-secret.png|cut -d' ' -f1`" ]
then
    echo "Extracting error (2)!";
    exit 0
fi


echo "------------------------------------------------------------------------"
echo "text + color png"
echo "------------------------------------------------------------------------"
rm -f testing/output-secret*
echo -n "Embed: "
time ./hstego.py embed testing/input-secret.txt testing/cover_color.png testing/stego_color.png p4ssw0rd
echo -n "Extract: "
time ./hstego.py extract testing/stego_color.png testing/output-secret.txt p4ssw0rd
if [ "`sha1sum testing/input-secret.txt|cut -d' ' -f1`" != "`sha1sum testing/output-secret.txt|cut -d' ' -f1`" ]
then
    echo "Extracting error (4)!";
    diff testing/input-secret.txt testing/output-secret.txt
    exit 0
fi




echo "------------------------------------------------------------------------"
echo "text + grayscale jpg"
echo "------------------------------------------------------------------------"
rm -f testing/output-secret*
echo -n "Embed: "
time ./hstego.py embed testing/input-secret-small.txt testing/cover.jpg testing/stego.jpg p4ssw0rd
echo -n "Extract: "
time ./hstego.py extract testing/stego.jpg testing/output-secret.txt p4ssw0rd
if [ "`sha1sum testing/input-secret-small.txt|cut -d' ' -f1`" != "`sha1sum testing/output-secret.txt|cut -d' ' -f1`" ]
then
    echo "Extracting error (3)!";
    diff testing/input-secret.txt testing/output-secret.txt
    exit 0
fi


echo "------------------------------------------------------------------------"
echo "text + color jpg"
echo "------------------------------------------------------------------------"
rm -f testing/output-secret*
echo -n "Embed: "
time ./hstego.py embed testing/input-secret-small2.txt testing/cover_color.jpg testing/stego_color.jpg p4ssw0rd
echo -n "Extract: "
time ./hstego.py extract testing/stego_color.jpg testing/output-secret.txt p4ssw0rd
if [ "`sha1sum testing/input-secret-small2.txt|cut -d' ' -f1`" != "`sha1sum testing/output-secret.txt|cut -d' ' -f1`" ]
then
    echo "Extracting error (5)!";
    diff testing/input-secret.txt testing/output-secret.txt
    exit 0
fi





