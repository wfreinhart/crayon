#
# list_test.py
# python script to provide list of unit tests to cmake
#
# Copyright (c) 2018 Wesley Reinhart.
# This file is part of the crayon project, released under the Modified BSD License.

import os
test_path = os.path.abspath(os.path.dirname(__file__))
import glob
for test in glob.glob(test_path + "/test_*.py"):
    print(test)
