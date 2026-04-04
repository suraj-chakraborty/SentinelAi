# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs

block_cipher = None

# Collect all submodules from the sentinel package
hidden_imports = collect_submodules('sentinel')
hidden_imports += [
    'mediapipe', 
    'cv2', 
    'pyttsx3.drivers', 
    'pyttsx3.drivers.sapi5',
    'fastapi',
    'uvicorn',
    'jinja2',
    'playwright',
    'chromadb',
    'sentence_transformers'
]

# Collect data files (templates, assets, etc.)
datas = [
    ('assets', 'assets'),
    ('sentinel/templates', 'sentinel/templates'),
    ('.env.example', '.'),
]

# Robustly find pvporcupine path for resource bundling
import pvporcupine
porcupine_path = os.path.dirname(pvporcupine.__file__)
datas += [
    (os.path.join(porcupine_path, 'resources'), 'pvporcupine/resources'),
    (os.path.join(porcupine_path, 'resources', 'keyword_files', 'windows'), 'pvporcupine/resources/keyword_files/windows'),
    (os.path.join(porcupine_path, 'lib', 'common'), 'pvporcupine/lib/common'),
    (os.path.join(porcupine_path, 'lib', 'windows'), 'pvporcupine/lib/windows'),
]
binaries = collect_dynamic_libs('pvporcupine')

# Collect mediapipe and sentence_transformers data
datas += collect_data_files('mediapipe')
datas += collect_data_files('sentence_transformers')

a = Analysis(
    ['sentinel_ai.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
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
    [],
    exclude_binaries=True,
    name='SentinelAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False, # Set to True if you want a console window for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico' if os.path.exists('assets/icon.ico') else None
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SentinelAI',
)
