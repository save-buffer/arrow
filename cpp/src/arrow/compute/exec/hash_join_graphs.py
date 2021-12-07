#!/bin/env python3

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import math
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

class Test:
    def __init__(self):
        self.times = []
        self.args = {}

    def get_argnames_by_cardinality_increasing(self):
        key = lambda x: len(set(self.args[x]))
        return sorted(self.args.keys(), key=key)

def organize_tests(filename):
    tests = {}
    with open(filename) as f:
        df = json.load(f)
        for idx, row in enumerate(df['benchmarks']):
            test_name = row['name']
            test_name_split = test_name.split('/')
            base_name = test_name_split[0]
            args = test_name_split[1:]
            if base_name not in tests.keys():
                tests[base_name] = Test()

            tests[base_name].times.append(row['ns/row'])
            
            if len(args) > 3:
                raise('Test can have at most 3 parameters! Found', len(args), 'in test', test_name)
            
            nonnamed_args = [x for x in args if ':' not in x]
            if len(nonnamed_args) > 1:
                raise('Test name must have only one non-named parameter! Found', len(nonnamed_args), 'in test', test_name)

            for arg in args:
                arg_name = ''
                arg_value = arg
                if ':' in arg:
                    arg_split = arg.split(':')
                    arg_name = arg_split[0]
                    arg_value = arg_split[1]
                if arg_name not in tests[base_name].args.keys():
                    tests[base_name].args[arg_name] = [arg_value]
                else:
                    tests[base_name].args[arg_name].append(arg_value)
    return tests;

def string_numeric_sort_key(val):
    try:
        return int(val)
    except:
        return str(val)

def plot_1d(test, argname, ax, label=None):
    x_axis = test.args[argname]
    y_axis = test.times
    ax.plot(x_axis, y_axis, label=label)
    ax.legend()
    ax.set_xlabel(argname)
    ax.set_ylabel('ns/row')

def plot_2d(test, sorted_argnames, ax, title):
    assert len(sorted_argnames) == 2
    lines = set(test.args[sorted_argnames[0]])
    ax.set_title(title)
    for line in sorted(lines, key=string_numeric_sort_key):
        indices = range(len(test.times))
        indices = list(filter(lambda i: test.args[sorted_argnames[0]][i] == line, indices))
        filtered_test = Test()
        filtered_test.times = [test.times[i] for i in indices]
        filtered_test.args[sorted_argnames[1]] = [test.args[sorted_argnames[1]][i] for i in indices]
        plot_1d(filtered_test, sorted_argnames[1], ax, '%s: %s' % (sorted_argnames[0], line))

def plot_3d(test, sorted_argnames):
    assert len(sorted_argnames) == 3
    num_graphs = len(set(test.args[sorted_argnames[0]]))
    num_rows = int(math.ceil(math.sqrt(num_graphs)))
    num_cols = int(math.ceil(num_graphs / num_rows))
    graphs = set(test.args[sorted_argnames[0]])

    for j, graph in enumerate(sorted(graphs, key=string_numeric_sort_key)):
        ax = plt.subplot(num_rows, num_cols, j + 1)
        filtered_test = Test()
        indices = range(len(test.times))
        indices = list(filter(lambda i: test.args[sorted_argnames[0]][i] == graph, indices))
        filtered_test.times = [test.times[i] for i in indices]
        filtered_test.args[sorted_argnames[1]] = [test.args[sorted_argnames[1]][i] for i in indices]
        filtered_test.args[sorted_argnames[2]] = [test.args[sorted_argnames[2]][i] for i in indices]
        plot_2d(filtered_test, sorted_argnames[1:], ax, '%s: %s' % (sorted_argnames[0], graph))

def main():
    if len(sys.argv) != 2:
        print('Usage: hash_join_graphs.py <data>.json')
        print('This script expects there to be a counter called ns/row as a field of every test in the JSON file.')
        return

    tests = organize_tests(sys.argv[1])

    for i, test_name in enumerate(tests.keys()):
        test = tests[test_name]
        sorted_argnames = test.get_argnames_by_cardinality_increasing()
        # Create a graph per lowest-cardinality arg
        # Create a line per second-lowest-cardinality arg
        # Use highest-cardinality arg as X axis
        fig = plt.figure(i)
        num_args = len(sorted_argnames)
        if num_args == 3:
            fig.suptitle(test_name)
            plot_3d(test, sorted_argnames)
            fig.subplots_adjust(hspace=0.4)
        elif num_args == 2:
            ax = plt.subplot()
            plot_2d(test, sorted_argnames, ax, test_name)
        else:
            ax = plt.subplot()
            plot_1d(test, sorted_argnames[0], ax)
        fig.set_size_inches(16, 9)
        fig.savefig('%s.svg' % test_name, dpi=fig.dpi, bbox_inches='tight')
        plt.show()

if __name__ == '__main__':
    main()
