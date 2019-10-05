from pathlib import Path
import sys

if 'win' in sys.platform:
    raise SystemExit('Ludwig does not support Windows')
elif 'linux' == sys.platform:
    mnt_point = '/media'
else:
    # assume MacOS
    mnt_point = '/Volumes'


class RemoteDirs:
    research_data = Path(mnt_point) / 'research_data'


class LocalDirs:
    root = Path(__file__).parent.parent
    src = root / 'ludwigviz'
    static = root / 'static'
    templates = root / 'templates'


class Default:
    field_input = 'default word'
    valid_type = 'term'

    project_name = 'CreateWikiCorpus'  # TODO only for dev
