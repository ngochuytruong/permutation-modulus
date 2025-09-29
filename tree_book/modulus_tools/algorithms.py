'''
Helpful algorithms for dealing with graphs and modulus.
'''

from itertools import combinations
import networkx as nx


def spanning_trees(G):
    '''
    Generates all spanning trees of a graph.
    
    Parameters
    ----------
    G : networkx graph
    
    Returns
    -------
    generator
        Each item returned is a list of edges for a spanning tree.
        The generator will produce every tree before terminating.
        
    Notes
    -----
    The algorithm looks at ALL combinations of |V|-1 edges in G and
    determines which are trees.  This will be very very slow on large
    graphs, so use with care.
    '''
        
    n = len(G.nodes)
    for T in combinations(G.edges,n-1):
        H = nx.Graph(T)
        if nx.is_tree(H):
            yield(T)