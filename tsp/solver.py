#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
from random import randint
import random
import numpy as np

Point = namedtuple("Point", ['x', 'y'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def cost_check(route, i, j, points):
    """Проверка на улучшение и реверс в случае успеха"""
    a, b, c, d = points[route[i-1]], points[route[i]], points[route[j-1]], points[route[j % len(route)]]

    d1 = length(a, c) + length(b, d)
    d0 = length(a, b) + length(c, d)

    delta = d1 - d0

    if delta >= 0:
        return 0

    route[i:j] = reversed(route[i:j])
    return delta

def swap(points, i, j):
    temp_point = points[i]
    points[i] = points[j]
    points[j] = temp_point
    return points

def join_all(n: int):
    """комбинации всех пар ребер"""
    return ((i, j)
            for i in range(n)
            for j in range(i + 2, n + (i > 0)))

def two_opt(route, points):
    # or_change = True

    all_pairs = join_all(len(route))

    while True:
        delta = 0

        for (a, b) in join_all(len(route)):
            delta += cost_check(route, a, b, points)
        if delta >= 0:
            break
    return route

def create_table(nodeCount, points):
    a = [[0] * (nodeCount) for i in range(nodeCount)]
    a = np.array(a)
    for i in range(nodeCount):
        for j in range(nodeCount):
            a[i][j] = length(points[i], points[j])
    return a

def find_next(index, d_m, all_points):
    """Ищем следующую ближайшую вершину"""
    min_path = 200000
    index_of_min = -1
    for j in all_points:
        if d_m[index][j] < min_path and j != index:
            min_path = d_m[index][j]
            index_of_min = j
    if index_of_min == -1:
        return index, []
    else:
        all_points.remove(index_of_min)
        return index_of_min, all_points

def greedy(s, distance_matrix, points):
    start_point = s
    all_points = list(range(len(distance_matrix[0])))
    all_points.remove(s)
    actual_point, all_points = find_next(start_point, distance_matrix, all_points)
    solution = []
    solution.append(start_point)
    solution.append(actual_point)

    i = 2
    while i < len(distance_matrix[0]):
        actual_point, all_points = find_next(actual_point, distance_matrix, all_points)
        solution.append(actual_point)
        i += 1
    return solution

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    solution = [i for i in range(nodeCount)]

    if nodeCount < 20000:
        # Greedy algorithm
        s = random.randint(0, nodeCount - 1)
        distance_matrix = create_table(nodeCount, points)
        solution = greedy(s, distance_matrix, points)
    else:
        solution = [i for i in range(nodeCount)]
        random.shuffle(solution)

    solution = two_opt(solution, points)

    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount - 1):
        obj += length(points[solution[index]], points[solution[index + 1]])

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

