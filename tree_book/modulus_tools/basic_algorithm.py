def matrix_modulus(N, p=2, sigma=None):
    """
    Computes the modulus of a family given its N matrix.
    
    Parameters
    ----------
    N : numpy array OR scipy sparse matrix
        The N matrix for the family.

    p : float or np.inf
        The modulus energy exponent.
        
    sigma : numpy array
        The weights sigma.  If sigma = None, all weights
        are treated as 1.
                
    Returns
    -------
    mod : float
        An approximation of the modulus.
        
    rho : numpy array
        An optimal rho^* for modulus.
        
    lam : numpy array
        An optimal dual lambda^* for modulus.
        
    Notes
    -----
    This function is just a template solver.  For specialized
    problems, it is probably best to implement a particular
    solver based on the energy exponent and the desired
    tolerances in the approximation.
    """
    
    import cvxpy as cvx
    import numpy as np

    from warnings import warn
    
    # problem dimension
    m = N.shape[1]
    
    # set sigma to default if necessary
    if sigma is None:
        sigma = np.ones(m)
    
    # convert inputs to cvxpy constants
    N = cvx.Constant(N)
    sigma = cvx.Constant(sigma)
    
    # primal variables
    rho = cvx.Variable(m)
    
    # objective
    if p is not np.inf:
        obj = cvx.Minimize(sigma.T@rho**p)
    else:
        obj = cvx.Minimize(cvx.max(cvx.multiply(sigma, rho)))
        
    # constraints
    cons = [rho >= 0, N@rho >= 1]
    
    # set up the problem
    prob = cvx.Problem(obj, cons)
    
    # attempt to solve
    prob.solve()
    if prob.status != 'optimal':
        warn('cvxpy solve returned status {}'.format(prob.status))
        
    return prob.value, np.array(rho.value).flatten(), np.array(cons[1].dual_value).flatten()

def modulus(m, solve_subproblem, find_shortest, p = 2, sigma = None, tol = 1e-3, max_iter = 1000,
            output_every = None):
    """
    Implements the basic algorithm for modulus.
    
    Parameters
    ----------
    m : int
        The dimension of the modulus problem (number of edges).
        
    solve_subproblem : callable
        See below.
        
    find_shortest : callable
        See below.

    p : float or np.inf
        The modulus energy exponent.
        
    sigma : numpy array
        The weights sigma.  If sigma = None, all weights
        are treated as 1.
        
    tol : float
        The tolerance.  The modulus algorithm stops when the approximate
        density is within tol of being admissible.
        
    max_iter : int
        Maximum number of iterations to perform before terminating with an error.
        
    output_every : int
        Frequency of output to stderr.  If this is set to None, output is supressed.
        
    Returns
    -------
    mod : float
        Approximation to modulus.
        
    cons : list
        List of constraints added during the iteration.  The format of the elements of
        this list is determined by the output of the find_violated_constraint function.
        
    rho : numpy array
        Approximation to an optimal density.
        
    lam : numpy array
        Approximation to the dual variables for the constraints listed in cons.
        
    Notes
    -----
    The function solve_subproblem should have the following signature
    
        mod, rho, lam = solve_subproblem(N, p, sigma)
        
    See the function matrix_modulus for an example.
    
    The function find_shortest should have the following signature
    
        cons, n = find_shortest(rho, tol)
        
    This function should find a "most violated constraint" using the specified values
    for rho.  Upon return, cons may contain any representation desired for describing
    the constraint.  (This is purely informational for the user and it is acceptable
    for cons to be set to None.)  n should contain the numpy row vector representing the
    violated constraint.  This is the row that will be added to N on the next iteration.
    
    The argument tol may be ignored.  However, it is acceptable for the function to return
    the tuple (None, None) if every constraint is satisfied to within a tolerance of tol.
    """
    
    import numpy as np
    import scipy.sparse as sp
    from time import perf_counter
    
    # timers
    search_time = 0.
    update_time = 0.
    mod_start = perf_counter()
    
    # initialize variables
    rho = np.zeros(m)
    N = sp.csr_matrix((0,m))
    lam = np.array([])
    mod = 0.
    upper = np.inf
    cons = []
    
    # default sigma
    if sigma is None:
        sigma = np.ones(m)
    
    # initialize output table
    if output_every:
        print('| {:>6s} | {:>9s} | {:>9s} | {:>9s} | {:>6s} | {:>9s} |' \
            .format( 'it', 'l bnd', 'u bnd', 'rel gap', '# cons', 'time (s)' ))
        print('+--------+-----------+-----------+-----------+--------+-----------+')

    # loop to at most max_iter
    for iter_count in range(max_iter):
                
        # find a constraint to add
        start = perf_counter()
        c, n = find_shortest(rho, tol)
        search_time += perf_counter()-start
        
        # compute the length of the shortest object
        if n is None:
            length = 1.
        else:
            length = n.dot(rho)
            
        # update the upper bound
        if length > 0:
            if p is np.inf:
                upper = np.abs(sigma*rho/length)
            else:
                upper = np.sum(sigma*(rho/length)**p)
            
        # check if we can stop
        if length > 1-tol:
            if output_every:
                print('| {:6d} | {:9.3e} | {:9.3e} | {:9.3e} | {:6d} | {:9.3e} |' \
                    .format( iter_count+1, mod, upper, (upper-mod)/mod, N.shape[0], perf_counter() - mod_start ))

                print()
                print('program running time = {} sec'.format( perf_counter() - mod_start ))
                print('constraint search    = {} sec'.format( search_time ))
                print('solution update      = {} sec'.format( update_time ))

            return mod, cons, rho, lam
        
        # if not, we need to add a constraint
        N = sp.vstack([N,n], format='csr')
        cons.append(c)
        
        # re-optimize
        start = perf_counter()
        mod, rho, lam =solve_subproblem(N, p, sigma)
        update_time += perf_counter()-start
        
        
        # print some feedback if desired
        if output_every and (iter_count+1) % output_every == 0:
            if mod == 0:
                rel_gap = np.inf
            else:
                rel_gap = (upper-mod)/mod
            print('| {:6d} | {:.3e} | {:.3e} | {:.3e} | {:6d} | {:.3e} |' \
                .format( iter_count+1, mod, upper, rel_gap, N.shape[0], perf_counter() - mod_start ))

        
    # if we got here, we failed to converge
    raise RuntimeError('Modulus algorithm failed to converge in {} iterations.'.format(max_iter))