#!/usr/bin/env python3

# Copyright 2017 Jeon.sungwook Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simple command-line example for Translate.

Command-line application that translates some text.
"""
from __future__ import print_function

__author__ = 'codetree@google.com (sungwook Jeon)'

import sys
import numpy as np
import matplotlib.pyplot as plt

def draw_plot(x_data, y_data):
    plt.plot(x_data, y_data, 'ro')
    plt.show()

def run_main(arg_param1):
    # [START run_main]
    num_points = 1000
    vectors_set =[]

    for i in range(num_points):
        x1 = np.random.normal(0.0, 0.55)
        y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
        vectors_set.append([x1, y1])

    x_data = [v[0] for v in vectors_set]
    y_data = [v[1] for v in vectors_set]

    draw_plot(x_data, y_data)
    print(arg_param1)
    # [END run_main]


if __name__ == '__main__':

    argc = len(sys.argv)
    if argc >= 2:
        print(sys.argv)
        # run_main(01_basic-operations.kr.srt)
        run_main(sys.argv[1])
    else:
        run_main("make data program")