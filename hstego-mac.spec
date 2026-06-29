# -*- mode: python ; coding: utf-8 -*-

import glob

from PyInstaller.utils.hooks import copy_metadata


block_cipher = None


def native_extension(name):
    candidates = sorted(glob.glob(f'build/lib.macosx*/{name}*'))
    if not candidates:
        raise FileNotFoundError(
            f"Native extension not found for {name}. Run python3 setup.py build first.")
    return candidates[-1]


a = Analysis(
    ['hstego.py'],
    pathex=[],
    binaries=[
       (native_extension('hstego_jpeg_toolbox_extension'), '.'), 
       (native_extension('hstego_stc_extension'), '.')
    ],
    datas=copy_metadata('imageio') + [('resources', 'resources')],
    hiddenimports=['PIL._tkinter_finder'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='hstego-0.6-macosx.universal',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
