import re
from bs4 import BeautifulSoup
from math import sqrt
from os import listdir
from os.path import isfile, join
import json
import seaborn as sns

#input_file_base = "./pydigest-dataset-master/data/pages/"


def text_density(input_file):
    html_string = open(input_file, errors="ignore").read()
    totallen = len(html_string)
    matches = re.findall('<.*?>', html_string)
    taglen = sum([len(m) for m in matches])
    textlen = len(BeautifulSoup(html_string, "html.parser").text)
    return textlen / totallen
