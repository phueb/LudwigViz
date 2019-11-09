import sys

if sys.platform == 'darwin':
    mnt_point = '/Volumes'
elif 'linux' == sys.platform:
    mnt_point = '/media'
else:
    raise SystemExit('Did not find mount point. User must add custom mount point')


__version__ = '1.0.0'

dummy_data = None  # overwrite if using dummy