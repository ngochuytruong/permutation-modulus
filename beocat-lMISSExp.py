import matplotlib.pyplot as plt
import numpy as np
import math
import cvxpy as cvx
import random
import multiprocess as mp
import warnings
import os
import sys


def generate_N(s):
    m_rows = []

    def recurse(s, x, row):
        row[x] = 1
        greater = []
        for i in range(x, len(s)):
            if s[i] > s[x] and all(s[j] > s[i] for j in greater):
                greater.append(i)
        if not greater:
            m_rows.append(row.copy())
            return
        for element in greater:
            recurse(s, element, row)
            row[element] = 0
            
    for i, element in enumerate(s):
        if all(element < x for x in s[:i]):
            recurse(s, i, np.zeros(len(s)))
        
    return np.array(m_rows)


def add_constraint(N, c):
    if N is None:
        return c
    return np.vstack((N, c))


def modulus(N, p):
    n = N.shape[-1]

    rho = cvx.Variable(n)

    cons = [rho >= 0, N@rho >= 1]

    obj = cvx.Minimize(cvx.pnorm(1**(1/p)*rho, p)**p)

    prob = cvx.Problem(obj, cons)
    prob.solve(solver=cvx.CLARABEL)

    return obj.value, np.array(rho.value).flatten()


def basic_matrix(pi, p):
    usage = generate_N(pi)
    rho = np.zeros(len(pi))
    mod = 0
    tol = 1e-5
    N = None
    longest_miss = 0
    while True:
        vals = usage @ rho           # shape (m,)

        # Find the smallest one
        idx_min = np.argmin(vals)
        min_value = vals[idx_min]
        row_min = usage[idx_min]
        size = 0
        for i in range(len(row_min)):
            if row_min[i] == 1:
                size = size + 1
        if size > longest_miss:
            longest_miss = size

        if min_value > 1 - tol:
            return mod, longest_miss
            # return mod, rho

        N = add_constraint(N, row_min)

        mod, rho = modulus(N, p)

    return mod


def compute_modulus(n):
    long_miss = []
    for _ in range(5000):
        val,longest_miss = basic_matrix(random.sample(range(1, n + 1), n), 1)
        long_miss.append(longest_miss)
    return sum(long_miss) / len(long_miss)


if __name__ == "__main__":
    start = 1
    stop = 50

    warnings.filterwarnings('ignore', category=FutureWarning)
    with mp.Pool(processes=64) as pool:
        results = pool.map(compute_modulus, range(start, stop))

    for n, result in zip(range(start, stop), results):
        print(f'{n}: {result}')

    #plt.title(f"Modulus Values of Permutations of Size {1} to {stop}")
    #plt.xlabel("Size of Permutation")
    #plt.ylabel("Value of Modulus")
    #plt.scatter(range(start, stop), results, label="Approximate Values")
    #plt.legend(loc="lower right")
    #plt.savefig("graph.png")