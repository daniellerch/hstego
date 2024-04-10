#!/bin/bash

rm -fr build/*
rm -f dist/*
python3 setup.py build
pyinstaller hstego-linux.spec

