import os
from setuptools import setup, Extension

m_jpg = Extension('jpeg_toolbox_extension', 
                  sources = ['src/jpeg_toolbox_extension.c'], 
                  libraries = ['jpeg'])

m_stc = Extension('stc_extension', 
                  include_dirs = ['src/'],
                  sources = ['src/common.cpp',
                             'src/stc_embed_c.cpp',
                             'src/stc_extract_c.cpp',
                             'src/stc_interface.cpp',
                             'src/stc_ml_c.cpp'],
                  extra_compile_args = ['-std=c++98'],
                  libraries = ['jpeg'])


here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()

setup(name = 'hstego',
      version = '1.0',
      author="Daniel Lerch",
      author_email="dlerch@gmail.com",
      url="https://github.com/daniellerch/hstego",
      description = 'Hard to detect image steganography',
      py_modules = ["hstego"],
      ext_modules = [m_jpg, m_stc])
