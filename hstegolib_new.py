#!/usr/bin/env python3


import base64
import hashlib

from ctypes import *
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

class Cipher:

    def __init__(self, password):
        self.password = password

    def open_file(self, path):
        with open(path, 'rb') as f:
            self.plaintext = f.read()

    def aes_encrypt(self):
        salt = get_random_bytes(AES.block_size)

        # use the Scrypt KDF to get a private key from the password
        private_key = hashlib.scrypt(
            self.password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)

        cipher = AES.new(private_key, AES.MODE_CBC)
        ciphertext = cipher.encrypt(pad(self.plaintext, AES.block_size))

        # Ciphertext with a 16+16 header 
        self.ciphertext = salt + cipher.iv + ciphertext

    def encrypt(self, input_path):
        """ encrypt file content """
        self.open_file(input_path)
        self.aes_encrypt()
        return self.ciphertext
  
    def aes_decrypt(self):
        salt = self.ciphertext[:AES.block_size]
        iv = self.ciphertext[AES.block_size:AES.block_size*2]
        ciphertext = self.ciphertext[AES.block_size*2:]

        # Fix padding
        mxlen = len(ciphertext)-(len(ciphertext)%AES.block_size)
        ciphertext = ciphertext[:mxlen]

        private_key = hashlib.scrypt(
            self.password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)

        cipher = AES.new(private_key, AES.MODE_CBC, iv=iv)
        decrypted = cipher.decrypt(ciphertext)
        self.decrypted = unpad(decrypted, AES.block_size)

    def decrypt(self, bytes_array, output_path):
        """ decrypt """
        self.ciphertext = bytes_array
        self.aes_decrypt()

        with open(output_path, 'wb') as f:
            f.write(bytes(self.decrypted))



path = "testing/input-secret-small.txt"
password = "p4ssw0rd"


cipher = Cipher(password)
bytes_array = cipher.encrypt(path)
#print(bits)


cipher.decrypt(bytes_array, "output.txt")










