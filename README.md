# HStego
HStego: Hard to detect image steganography.

HStego is a tool for hiding data in bitmap and JPEG images.
This tool uses some of the most advanced steganography methods known today, along with an upper limit on the amount of data that can be hidden so that it cannot be reliably detected by modern steganography tools.



## Install
```bash 
pip3 install imageio scipy pycryptodome
pip3 install git+https://github.com/daniellerch/hstego
```

## Uninstall
```bash 
pip3 uninstall hstego
```

## Other useful commands

### Build only
```bash 
python3 setup.py build
```

### Create wheel
```bash 
python3 setup.py bdist_wheel
```

### Install whl
```bash 
pip3 install dist/hstego-1.0-cp37-cp37m-linux_x86_64.whl
```




## Command line examples:

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


            



