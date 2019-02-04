import os
import re


def mkdir_p(adir):
    """Recursively create a directory"""
    adir = os.path.abspath(adir)
    while not os.path.exists(adir):
        try:
            os.mkdir(adir)
        except:
            mkdir_p(os.path.dirname(adir))


def alphanum2num(string):
    """Removes all non-numeric characters from a string and returns a float"""
    try:
        ret = float(re.sub("[^0-9]", "", string))
    except:
        print(string)
        ret = ""
    return ret
