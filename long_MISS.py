import matplotlib.pyplot as plt
import numpy as np
import math
import cvxpy as cvx
import random
import warnings
import os
import sys
import multiprocess as mp

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

def compute_long_MISS(n):

    MISS_array = []
    for _ in range(5000):
        pi = random.sample(range(1, n + 1), n)
        usage = generate_N(pi)
        longest = 0
        for i in range(len(usage)):
            count = 0
            for j in range(len(usage[i])):
                if usage[i][j] == 1:
                    count = count + 1
            if count > longest:
                longest = count
        MISS_array.append(longest)

    return sum(MISS_array) / len(MISS_array)

if __name__ == "__main__":
    start = 1
    stop = 50
        
    warnings.filterwarnings('ignore', category=FutureWarning)
    with mp.Pool(processes=64) as pool:
        results = pool.map(compute_long_MISS, range(start, stop))

    for n, result in zip(range(start, stop), results):
        print(f'{n}: {result}')

    #plt.title(f"Modulus Values of Permutations of Size {1} to {stop}")
    #plt.xlabel("Size of Permutation")
    #plt.ylabel("Value of Modulus")
    #plt.scatter(range(start, stop), results, label="Approximate Values")
    #plt.legend(loc="lower right")
    #plt.savefig("graph.png")