import os
from setuptools import setup, Extension
from distutils import sysconfig

sysconfig.get_config_vars()['CC'] = 'emcc'
sysconfig.get_config_vars()['CXX'] = 'emcc'
sysconfig.get_config_vars()['LDSHARED'] = 'emcc'


m_jpg = Extension('hstego_jpeg_toolbox_extension', 
                  sources = ['src/jpeg_toolbox_extension.c'], 
                  libraries = ['jpeg'])

m_stc = Extension('hstego_stc_extension', 
                  include_dirs = ['src/'],
                  sources = ['src/common.cpp',
                             'src/stc_embed_c.cpp',
                             'src/stc_extract_c.cpp',
                             'src/stc_interface.cpp',
                             'src/stc_ml_c.cpp'],
                  extra_compile_args = ['-s', 'WASM=1', '-O2', '-std=c++11', '-Wno-narrowing'],
                  )



here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()

setup(name = 'hstego',
      version = '0.4',
      author="Daniel Lerch",
      author_email="dlerch@gmail.com",
      url="https://github.com/daniellerch/hstego",
      description = 'Hard to detect image steganography',
      py_modules = ["hstegolib", "hstegogui"],
      scripts = ['hstego.py'],
      data_files = [('resources', ['resources/hide.png']),
                    ('resources', ['resources/extract.png'])],
      ext_modules = [m_jpg, m_stc])
