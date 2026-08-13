# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['src/fast_nc_zarr/application/desktop_worker/sidecar_main.py'],
    pathex=['src'],
    binaries=[('/run/media/owen/HDD/zarr-fast-converter-v1/src/fast_nc_zarr/_native.cpython-313-x86_64-linux-gnu.so', 'fast_nc_zarr')],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='fast-nc-zarr-worker',
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
