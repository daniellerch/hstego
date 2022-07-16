import os
from setuptools import setup, Extension
from distutils.command.build_ext import build_ext as build_ext_orig


from distutils.command.build_ext import build_ext as build_ext_orig
class CTypesExtension(Extension): pass
class build_ext(build_ext_orig):

    def build_extension(self, ext):
        self._ctypes = isinstance(ext, CTypesExtension)
        return super().build_extension(ext)

    def get_export_symbols(self, ext):
        if self._ctypes:
            return ext.export_symbols
        return super().get_export_symbols(ext)

    def get_ext_filename(self, ext_name):
        if self._ctypes:
            return ext_name + '.so'
        return super().get_ext_filename(ext_name)


m_stc = CTypesExtension('hstego_stc_extension', 
                  include_dirs = ['src/'],
                  sources = ['src/stc_interface.cpp',
                             'src/common.cpp',
                             'src/stc_embed_c.cpp',
                             'src/stc_extract_c.cpp',
                             'src/stc_ml_c.cpp'],
                  )


m_jpg = CTypesExtension('hstego_jpeg_toolbox_extension', 
                  sources = ['src/jpeg_toolbox_extension.c'], 
                  include_dirs = ['src/jpeg-9c-win/Include/'],
                  libraries = ['src/jpeg-9c-win/Lib/static_x64/jpeg'],
                  )


here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()

setup(name = 'hstego',
      version = '0.3',
      author="Daniel Lerch",
      author_email="dlerch@gmail.com",
      url="https://github.com/daniellerch/hstego",
      description = 'Hard to detect image steganography',
      py_modules = ["hstegolib", "hstegogui"],
      scripts = ['hstego.py'],
      data_files = [('resources', ['resources/hide.png']),
                    ('resources', ['resources/extract.png'])],
      ext_modules = [m_stc, m_jpg],
      cmdclass={'build_ext': build_ext})


