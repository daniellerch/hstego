# HStego
**HStego:** Hard to detect image steganography.

HStego is a tool for hiding data in bitmap and JPEG images.
This tool uses some of the most advanced steganography methods known today, along with an upper limit on the amount of data that can be hidden so that it cannot be reliably detected by modern steganography tools.


> **WARNING:** This tool is in a ALPHA stage. Use at your own risk.
> If you find any problem, please open a [issue](https://github.com/daniellerch/hstego/issues).


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

# Hide a message
hstegolib.HILL_embed("cover.png", "secret.txt", "MyP4ssw0rd101", "stego.png")

# Extract the message
hstegolib.HILL_extract("stego.png", "MyP4ssw0rd101", "content.txt")
```


Example using JPEG images:

```python
import hstegolib

# Hide a message
hstegolib.J_UNIWARD_embed("cover.jpg", "secret.txt", "MyP4ssw0rd101", "stego.jpg")

# Extract the message
hstegolib.J_UNIWARD_extract("stego.png", "MyP4ssw0rd101", "content.txt")
```


## Acknowledgments:

HStego implements the J-UNIWARD method for JPEG images and the HILL method for bitmap images. These
methods are described in the following papers:


- Universal Distortion Function for Steganography in an Arbitrary Domain by Vojtěch Holub, Jessica 
  Fridrich and Tomáš Denemark

- A New Cost Function for Spatial Image Steganography by Bin Li, Ming Wang, Jiwu Huang and Xiaolong Li.


Part of the C/C++ code used by HStego comes from the [Digital Data Embedding Laboratory](http://dde.binghamton.edu/download/).

This software would not have been possible without their excellent work.





