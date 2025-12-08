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
    while True:
        vals = usage @ rho           # shape (m,)

        # Find the smallest one
        idx_min = np.argmin(vals)
        min_value = vals[idx_min]
        row_min = usage[idx_min]

        if min_value > 1 - tol:
            return mod
            # return mod, rho

        N = add_constraint(N, row_min)

        mod, rho = modulus(N, p)

    return mod


def compute_modulus(n):
    mod = 0
    mod_ext = 0
    sum = 0
    probs = []
    for i in range(n+1):
        pi = random.sample(range(1, n + 1), n)
        mod = basic_matrix(pi, 1)
        new_val = i
        for a in len(pi):
            if pi[a] >= new_val:
                pi[a] = 1 + pi[a]
        pi.append(new_val)
        mod_ext = basic_matrix(pi, 1)
        if mod_ext > mod:
            probs.append(1)
        else:
            probs.append(0)
    return probs


if __name__ == "__main__":
    start = 70
    stop = 71

    full_probs = []
    for i in range(71):
        full_probs.append(0)

    for _ in range(5000):
        prob = compute_modulus(start)
        for i in prob:
            if prob[i] == 1:
                full_probs[i] = full_probs[i] + 1

    for i in range(71):
        full_probs[i] = full_probs[i] / 5000

    for n, result in zip(range(1,71), full_probs):
        print(f'{n}: {result}')

    #plt.title(f"Modulus Values of Permutations of Size {1} to {stop}")
    #plt.xlabel("Size of Permutation")
    #plt.ylabel("Value of Modulus")
    #plt.scatter(range(start, stop), results, label="Approximate Values")
    #plt.legend(loc="lower right")
    #plt.savefig("graph.png")