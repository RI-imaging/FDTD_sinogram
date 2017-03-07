#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals

import os

def mkdir_p(adir):
    """Recursively create a directory"""
    adir = os.path.abspath(adir)
    while not os.path.exists(adir):
        try:
            os.mkdir(adir)
        except:
            mkdir_p(os.path.dirname(adir))