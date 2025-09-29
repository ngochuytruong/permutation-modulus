"""
Operator functors on families.
"""

import numpy as np

class UnionShortest:
    """
    Shortest object operator for a union of families.
    """
    
    def __init__(self, F ):
        self.F = F
        
    def __call__(self, rho, tol):
        
        results = [f(rho, tol) for f in self.F]
        lengths = [rho.dot(n) for cons, n in results]
        ind = np.argmin(lengths)
        return results[ind]
    
class SumShortest:
    """
    Shortest object operator for a summation of families.
    """

    def __init__(self, F ):
        self.F = F
        
    def __call__(self, rho, tol):
        
        n = np.zeros(rho.shape)
        cons = []
        for f in self.F:
            c_f, n_f = f(rho, tol)
            cons.append(c_f)
            n += n_f
        
        return cons, n
