import datetime
import re

from ludwigviz import config

regex_digit = re.compile(r'[0-9]+')


def sort_rows(rows, header, order):

    assert header in rows[0]  # make sure that the header is actually in use

    if header == 'Last Modified':
        print('Sorting using datetime')
        res = sorted(rows,
                     key=lambda row: datetime.datetime.strptime(row[header], config.Time.format),
                     reverse=True if order == 'descending' else False)
    else:
        res = sorted(rows,
                     key=lambda row: row[header],
                     reverse=True if order == 'descending' else False)
    return res


def to_param_id(param_name):
    return regex_digit.search(param_name).group()


def get_time_modified(p):
    return datetime.datetime.fromtimestamp(
        p.lstat().st_mtime).strftime(config.Time.format)