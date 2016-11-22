#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals

import os
import sys
import warnings

def mkdir(newdir):
    """works like "mkdir -p"
        - already exists, silently complete
        - regular file in the way, raise an exception
        - parent directory(ies) does not exist, make them as well
    """
    newdir = os.path.abspath(newdir)
    if os.path.isdir(newdir):
        pass
    elif os.path.isfile(newdir):
        raise OSError("a file with the same name as the desired " \
                      "dir, '%s', already exists." % newdir)
    else:
        head, tail = os.path.split(newdir)
        if head and not os.path.isdir(head):
            mkdir(head)
        #print "_mkdir %s" % repr(newdir)
        if tail:
            try:
                os.mkdir(newdir)
            except:
                warnings.warn("Could not create dir: \n{}".format(
                                                     sys.exc_info()[1]))
        
