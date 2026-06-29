import os
import subprocess
from setuptools import setup, Extension


def jpeg_dirs():
      local_include = os.path.join('src', 'jpeg-9c-win', 'Include')
      try:
            prefix = subprocess.check_output(
                  ['brew', '--prefix', 'jpeg'], text=True).strip()
      except (OSError, subprocess.CalledProcessError):
            return [local_include], []
      return [os.path.join(prefix, 'include'), local_include], [os.path.join(prefix, 'lib')]


jpeg_includes, jpeg_libraries = jpeg_dirs()


m_jpg = Extension('hstego_jpeg_toolbox_extension', 
                  include_dirs = jpeg_includes,
                  sources = ['src/jpeg_toolbox_extension.c'], 
                  library_dirs = jpeg_libraries,
                  libraries = ['jpeg'])

m_stc = Extension('hstego_stc_extension', 
                  include_dirs = ['src/'],
                  sources = ['src/common.cpp',
                             'src/stc_embed_c.cpp',
                             'src/stc_extract_c.cpp',
                             'src/stc_interface.cpp',
                             'src/stc_ml_c.cpp'],
                  extra_compile_args = ['-std=c++11', '-Wno-narrowing'],
                  )



here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()

setup(name = 'hstego',
      version = '0.6',
      author="Daniel Lerch",
      author_email="dlerch@gmail.com",
      url="https://github.com/daniellerch/hstego",
      description = 'Hard to detect image steganography',
      py_modules = ["hstegolib", "hstegogui"],
      scripts = ['hstego.py'],
      data_files = [('resources', ['resources/hide.png']),
                    ('resources', ['resources/extract.png'])],
      ext_modules = [m_jpg, m_stc])
