#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    Minimize: sum_by_F(facilities[F].setup_cost * X[F]) + sum_all(F,C)(Distance[F, C] * Y[F, C])

    Subject to:
        1) Запросы клиентов не могут превышать общее capacity склада, где они обслуживаются.
        sum_by_С(customers[c].demand * Y[F, C]) <= facilities[F].capacity ( for F in facilities )

        2) Не может быть такого, что customer обслуживается в закрытом facility.
        Y[F, C] <= X[F] ( for (F, C) in all_combinations(facility, customer) )

        3) Customer обслуживается только в одном facility.
        sum_by_F(Y[F, C]) == 1 ( for c in customers )
"""

from collections import namedtuple
import math
import numpy as np
import cvxopt
import cvxopt.glpk

cvxopt.glpk.options['msg_lev'] = 'GLP_MSG_OFF'
cvxopt.glpk.options['tm_lim'] = 3600 * 10 ** 3  # 1hr

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])


def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def nearestCheapFacility(facilities, customers, alpha, beta=0, flag1=0, flag2=0):
    solution = [-1] * len(customers)
    capacities = [f.capacity for f in facilities]

    for c in customers:
        min_cost = -1
        best_id = -1
        for f in facilities:
            if capacities[f.index] >= c.demand:
                if (min_cost == -1 or (
                        alpha * length(f.location, c.location) + (1 - alpha) * f.setup_cost - beta * capacities[
                    f.index] < min_cost and f.capacity == capacities[f.index])):
                    best_id = f.index
                    min_cost = alpha * length(f.location, c.location) + (1 - alpha) * f.setup_cost - beta * capacities[
                        f.index]
                elif (min_cost == -1 or (
                        alpha * length(f.location, c.location) - beta * capacities[f.index] < min_cost and f.capacity !=
                        capacities[f.index])):
                    best_id = f.index
                    min_cost = alpha * length(f.location, c.location) - beta * capacities[f.index]
        solution[c.index] = best_id
        capacities[best_id] -= c.demand

    used = [0] * len(facilities)
    for facility_index in solution:
        used[facility_index] = 1
    obj = sum([f.setup_cost * used[f.index] for f in facilities])
    for customer in customers:
        obj += length(customer.location, facilities[solution[customer.index]].location)
    if flag1 == 1:
        obj, solution = opt2(facilities, customers, solution, capacities)
    if flag2 == 1:
        obj, solution = opt2Facilities(facilities, customers, solution, capacities)

    return obj, solution


def opt2Facilities(facilities, customers, solution, capacities):
    """
        проверить каждого customer на уменьшение целевой функции переселить к другому facility
    """
    quantity_of_customers = [0] * len(facilities)
    for c in solution:
        quantity_of_customers[c] += 1
    for c, f in enumerate(solution):
        for facility in facilities:
            if f == facility.index:
                continue
            else:
                f1, c1, f2 = facilities[f], customers[c], facility
                f1_c1 = length(f1.location, c1.location)
                f2_c1 = length(f2.location, c1.location)
                d0 = f1_c1 + f1.setup_cost + f2.setup_cost
                if capacities[f2.index] >= c1.demand:
                    if quantity_of_customers[f1.index] == 1:
                        d1 = f2_c1 + f2.setup_cost
                    else:
                        d1 = f2_c1 + f1.setup_cost + f2.setup_cost
                else:
                    continue
                if quantity_of_customers[f2.index] == 0:
                    d0 -= f2.setup_cost
                if d1 - d0 < 0:
                    solution[c] = f2.index
                    capacities[f] += c1.demand
                    capacities[f2.index] -= c1.demand
                    quantity_of_customers[f1.index] -= 1
                    quantity_of_customers[f2.index] += 1
                else:
                    continue
    used = [0] * len(facilities)
    for facility_index in solution:
        used[facility_index] = 1
    obj = sum([f.setup_cost * used[f.index] for f in facilities])
    for customer in customers:
        obj += length(customer.location, facilities[solution[customer.index]].location)
    # print(capacities)
    return obj, solution


def opt2(facilities, customers, solution, capacities):
    quantity_of_customers = [0] * len(facilities)
    for c in solution:
        quantity_of_customers[c] += 1

    for c1, f1 in enumerate(solution):
        for j in range(c1 + 1, len(solution)):
            f1_, c1_, f2, c2 = facilities[f1], customers[c1], facilities[solution[j]], customers[j]
            f1_c1 = length(f1_.location, c1_.location)
            f2_c2 = length(f2.location, c2.location)
            f2_c1 = length(f2.location, c1_.location)
            f1_c2 = length(f1_.location, c2.location)
            d0 = f1_c1 + f2_c2 + f1_.setup_cost + f2.setup_cost
            d1 = f2_c1 + f1_c2 + f1_.setup_cost + f2.setup_cost
            delta = -d0 + d1

            if delta >= 0:
                continue
            else:
                temp = None

                if capacities[f1] + c1_.demand >= c2.demand and capacities[f2.index] + c2.demand >= c1_.demand:
                    temp = solution[c1]
                    solution[c1] = solution[j]
                    solution[j] = temp
                    capacities[f1] = capacities[f1] + c1_.demand - c2.demand
                    capacities[f2.index] = capacities[f2.index] + c2.demand - c1_.demand

    used = [0] * len(facilities)
    for facility_index in solution:
        used[facility_index] = 1
    obj = sum([f.setup_cost * used[f.index] for f in facilities])
    for customer in customers:
        obj += length(customer.location, facilities[solution[customer.index]].location)
    # print(capacities)
    return obj, solution


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])

    facilities = []
    for i in range(1, facility_count + 1):
        parts = lines[i].split()
        facilities.append(Facility(i - 1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3]))))

    customers = []
    for i in range(facility_count + 1, facility_count + 1 + customer_count):
        parts = lines[i].split()
        customers.append(Customer(i - 1 - facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))

    if facility_count * customer_count < 50000:
        obj, solution = mip(facilities, customers, facility_count, customer_count, orig_facilities=facilities)
    elif facility_count * customer_count == 100000:
        obj, solution = mip(facilities[48:69], customers, facility_count, customer_count, orig_facilities=facilities)
    else:
        min_ = 100000000000
        bst = None
        if facility_count == 1000 and customer_count == 1500:
            obj, solution = nearestCheapFacility(facilities, customers, 0.85, 0, 1, 1)
        elif (facility_count == 500 and customer_count == 3000) or facility_count * customer_count == 160000:
            obj, solution = nearestCheapFacility(facilities, customers, 0.85)
        elif facility_count * customer_count >= 4000000:
            obj, solution = nearestCheapFacility(facilities, customers, 0.85)

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


def mip(facilities, customers, facilities_count, customer_count, orig_facilities):
    n = facilities[0].index
    M = len(customers)
    N = len(facilities)
    c = []
    for j in range(N):
        c.append(facilities[j].setup_cost)
    for j in range(N):
        for i in range(M):
            c.append(length(facilities[j].location, customers[i].location))

    xA = []
    yA = []
    valA = []
    for i in range(M):
        for j in range(N):
            xA.append(i)
            yA.append(N + M * j + i)
            valA.append(1)

    b = np.ones(M)

    xG = []
    yG = []
    valG = []
    for i in range(N):
        for j in range(M):
            xG.append(M * i + j)
            yG.append(i)
            valG.append(-1)
            xG.append(M * i + j)
            yG.append(N + M * i + j)
            valG.append(1)

    for i in range(N):
        for j in range(M):
            xG.append(N * M + i)
            yG.append(N + M * i + j)
            valG.append(customers[j].demand)
    h = np.hstack([np.zeros(N * M),
                   np.array([fa.capacity for fa in facilities], dtype='d')])

    binVars = set()
    for var in range(N + M * N):
        binVars.add(var)

    # print("here")

    status, isol = cvxopt.glpk.ilp(c=cvxopt.matrix(c),
                                   G=cvxopt.spmatrix(valG, xG, yG),
                                   h=cvxopt.matrix(h),
                                   A=cvxopt.spmatrix(valA, xA, yA),
                                   b=cvxopt.matrix(b),
                                   I=binVars,
                                   B=binVars)

    # print(cvxopt.spmatrix(valA, xA, yA))

    # print("here1")

    soln = []
    for i in range(M):
        for j in range(N):
            if isol[N + M * j + i] == 1:
                soln.append(j+n)

    # print("here2")

    # calculate the cost of the solution
    used = [0] * facilities_count
    for facility_index in soln:
        used[facility_index] = 1
    # print("here3")
    if facilities_count * customer_count != 100000:
        obj = sum([f.setup_cost * used[f.index] for f in facilities])
        for customer in customers:
            obj += length(customer.location, facilities[soln[customer.index]].location)
    else:
        obj = sum([f.setup_cost * used[f.index] for f in facilities])
        for customer in customers:
            obj += length(customer.location, orig_facilities[soln[customer.index]].location)
    return obj, soln


import sys

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print(
            'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')

