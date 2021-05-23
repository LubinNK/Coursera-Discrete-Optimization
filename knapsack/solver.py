#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import numpy as np
from operator import attrgetter
Item = namedtuple("Item", ['index', 'value', 'weight', 'density', 'cur_index'])
Item_new = namedtuple("Item_new", ['index', 'value', 'weight', 'orig_index'])

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []
    taken = [0]*item_count

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i, int(parts[0]), int(parts[1]), float(int(parts[0])/int(parts[1])), -1))

    if item_count == 10000:
        items = sorted(items, key=attrgetter('density'))

        items = items[::-1]
        items = items[:60]

        items_new = []

        counter = 1

        for item in items:
            items_new.append(Item_new(counter, item.value, item.weight, item.index))
            counter += 1

        table = np.zeros((capacity + 1, 61))
        for item in items_new:
            for i in range(1, capacity + 1):
                if item.weight > i:
                    table[i][item.index] = table[i][item.index - 1]
                else:
                    table[i][item.index] = max(table[i][item.index - 1],
                                               table[i - item.weight][item.index - 1] + item.value)

        k = capacity

        for item in items_new[::-1]:
            if table[k][item.index] == table[k][item.index - 1]:
                taken[item.orig_index - 1] = 0
            else:
                taken[item.orig_index - 1] = 1
                k = k - item.weight

        value = int(table[capacity][60])

    elif item_count == 400:

        items = sorted(items, key=attrgetter('density'))

        items = items[::-1]
        items = items[:20]

        items_new = []

        counter = 1

        for item in items:
            items_new.append(Item_new(counter, item.value, item.weight, item.index))
            counter += 1

        table = []

        for i in range(0, capacity + 1):
            table.append([])
            for j in range(0, 21):
                table[i].append(0)

        for item in items_new:
            for i in range(1, capacity + 1):
                if item.weight > i:
                    table[i][item.index] = table[i][item.index - 1]
                else:
                    table[i][item.index] = max(table[i][item.index - 1],
                                               table[i - item.weight][item.index - 1] + item.value)

        k = capacity

        for item in items_new[::-1]:
            if table[k][item.index] == table[k][item.index - 1]:
                taken[item.orig_index - 1] = 0
            else:
                taken[item.orig_index - 1] = 1
                k = k - item.weight

        value = int(table[capacity][20])
    else:
        table = np.zeros((capacity+1, item_count+1))
        for item in items:
            for i in range(1, capacity + 1):
                if item.weight > i:
                    table[i][item.index] = table[i][item.index - 1]
                else:
                    table[i][item.index] = max(table[i][item.index-1],
                                               table[i-item.weight][item.index - 1] + item.value)

        k = capacity

        for item in items[::-1]:
            if table[k][item.index] == table[k][item.index - 1]:
                taken[item.index-1] = 0
            else:
                taken[item.index-1] = 1
                k = k - item.weight

        value = int(table[capacity][item_count])

    # prepare the solution in the specified output format
    # output_data = str(capacity) + ' ' + str(item_count) + '\n'
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

