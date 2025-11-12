"""Utility: compile all .py files in the repo to find syntax errors.

Run from the repo root:
    python tools/py_compile_all.py

Exits with non-zero status if any file fails to compile.
"""
import py_compile
import glob
import sys

files = glob.glob('**/*.py', recursive=True)
# Skip virtualenv and common large folders
skipped_prefixes = ('venv', '.venv', '__pycache__', 'pretrained_models', 'runtime')
errs = 0
for f in files:
    if any(p in f.split('/') or p in f.split('\\') for p in skipped_prefixes):
        continue
    try:
        py_compile.compile(f, doraise=True)
    except Exception as e:
        print(f'FAIL {f}: {e}')
        errs = 1

if errs:
    print('\npy_compile finished: FAIL')
    sys.exit(1)
else:
    print('\npy_compile finished: OK')
    sys.exit(0)
