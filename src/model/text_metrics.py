"""
Functions to compute various geometric properties of html files.
Probably this is a placeholder.
"""

import re
from keyword import kwlist

from bs4 import BeautifulSoup

# TODO: performance is stupidly low right now
kwlist.extend("= [ ] ( ) : , + - * /".split())


def get_text_density(html_string):
    totallen = len(html_string)
    textlen = len(BeautifulSoup(html_string, "html.parser").text)
    return textlen / totallen


def get_totallen(html_string):
    return len(html_string)


def get_taglen(html_string):
    matches = re.findall("<.*?>", html_string)
    return sum([len(m) for m in matches])


def get_textlen(html_string):
    textlen = len(BeautifulSoup(html_string, "html.parser").text)
    return textlen


def get_keyword_frequency(html_string):
    text = BeautifulSoup(str(html_string), "html.parser").text
    if len(text) == 0:
        return 0
    return sum(text.count(kword) for kword in kwlist)
