import sys

if 'win' in sys.platform:
    raise SystemExit('Ludwig does not support Windows')
elif 'linux' == sys.platform:
    mnt_point = '/media'
else:
    # assume MacOS
    mnt_point = '/Volumes'


__version__ = '1.0.0'

dummy_data = None  # overwrite if using dummy