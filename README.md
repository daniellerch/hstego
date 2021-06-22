## What is HStego?

**HStego:** Hard to detect image steganography.

HStego is a tool for hiding data in bitmap and JPEG images.
This tool uses some of the most advanced steganography methods known today, along with an upper limit on the amount of data that can be hidden so that it cannot be reliably detected by modern steganography tools.


> **WARNING:** This tool is in a ALPHA stage. Use at your own risk. <br>
> If you find any problem, please open a [issue](https://github.com/daniellerch/hstego/issues).



- [Install](#install)
- [Command line examples](#command-line-examples)
- [Python examples](#python-examples)
- [Technical details](technical-details)
- [Acknowledgments](#acknowledgments)


## Install

You can install HStego with the following commands:
```bash 
pip3 install imageio scipy pycryptodome
pip3 install git+https://github.com/daniellerch/hstego
```

Uninstall with:
```bash 
pip3 uninstall hstego
```



## Command line examples:

HStego is a command line tool. Here are some examples of how to use it.


Example using bitmap images:

```bash
hstego.py embed secret.txt cover.png stego.png MyP4ssw0rd101
```

```bash
hstego.py extract stego.png content.txt MyP4ssw0rd101
```



Example using JPEG images:

```bash
hstego.py embed secret.txt cover.jpg stego.jpg MyP4ssw0rd101
```

```bash
hstego.py extract stego.jpg content.txt MyP4ssw0rd101
```




## Python examples

HStego can also be used as a Python library. Check out the following examples:

Example using bitmap images:

```python
import hstegolib
hill = hstegolib.HILL()

# Hide a message
hill.embed("cover.png", "secret.txt", "MyP4ssw0rd101", "stego.png")

# Extract the message
hill.extract("stego.png", "MyP4ssw0rd101", "content.txt")
```


Example using JPEG images:

```python
import hstegolib
juniw = hstegolib.J_UNIWARD()

# Hide a message
juniw.embed("cover.jpg", "secret.txt", "MyP4ssw0rd101", "stego.jpg")

# Extract the message
juniw.extract("stego.png", "MyP4ssw0rd101", "content.txt")
```


## Technical details:

HStego can hide information in bitmap and JPEG images. For hiding data into
bitmap images, HStego uses the HILL [1] cost function with Syndrome Trellis
Codes. 


## Acknowledgments:

HStego implements the J-UNIWARD method for JPEG images and the HILL method for bitmap images. These
methods are described in the following papers:


- [Universal Distortion Function for Steganography in an Arbitrary Domain](https://doi.org/10.1186/1687-417X-2014-1) by Vojtěch Holub, Jessica Fridrich and Tomáš Denemark

- [A New Cost Function for Spatial Image Steganography](https://doi.org/10.1109/ICIP.2014.7025854) by Bin Li, Ming Wang, Jiwu Huang and Xiaolong Li.


The embedding is implemeted using Syndrome Trellis Codes:

- [Minimizing Embedding Impact in Steganography using Trellis-Coded Quantization](https://doi.org/10.1117/12.838002) by Tomas Filler, Jan Judas and Jessica Fridrich.

- [Minimizing Additive Distortion Functions With Non-Binary Embedding Operation in Steganography](https://doi.org/10.1109/WIFS.2010.5711444) by Tomas Filler and Jessica Fridrich.

- [Minimizing Additive Distortion in Steganography using Syndrome-Trellis Codes](https://doi.org/10.1109/TIFS.2011.2134094) by Tomas Filler, Jan Judas and Jessica Fridrich.


Part of the C/C++ code used by HStego comes from the [Digital Data Embedding Laboratory](http://dde.binghamton.edu/download/).

This software would not have been possible without their excellent work.





