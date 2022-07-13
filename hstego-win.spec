# -*- mode: python ; coding: utf-8 -*-

import os

block_cipher = None


a = Analysis(
    ['hstego.py'],
    pathex=[],
    binaries=[
       ('build\lib.win-amd64-3.10\hstego_jpeg_toolbox_extension.so', '.'), 
       ('build\lib.win-amd64-3.10\hstego_stc_extension.so', '.')
    ],
    datas=[],
    hiddenimports=[],
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
    name='hstego-win.amd64',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
