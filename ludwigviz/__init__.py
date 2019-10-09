import sys

if sys.platform == 'darwin':
    mnt_point = '/Volumes'
elif 'linux' == sys.platform:
    mnt_point = '/media'
else:
    raise SystemExit('Ludwig currently does not support this platform')


__version__ = '1.0.0'

dummy_data = None  # overwrite if using dummy