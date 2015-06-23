# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import matplotlib
matplotlib.use('Agg')  #ugly hack to allow plotting from terminal
import matplotlib.pyplot as plt
import random
from copy import deepcopy

def _dist(p,q):
    return sum([(p[i]-q[i])**2 for i in xrange(len(p))])

def _lloyds_single_run(X, K, max_iterations, init):
    # Initialize with a subset of the data
    if init == 'sample':
        initials = random.sample(X, K)
    # Or initialize with random points across the same range as data
    elif init == 'scatter':
        vals = zip(*X)
        xmin = min(vals[0])
        xmax = max(vals[0])
        ymin = min(vals[1])
        ymax = max(vals[1])
        initials = [(random.uniform(xmin, xmax),
                     random.uniform(ymin, ymax)) for _ in xrange(K)]
    # Or yell RTFM at user
    else:
        raise ValueError('Invalid initialization mode!')

    #Contruct hashmap mapping integers up to K to centroids
    centroids = dict(enumerate(initials))
    converged = False
    iterations = 0
    
    while not converged and iterations < max_iterations:
        clusters = {i : [] for i in xrange(K)}
        #Make sure clusters and centroids have identical keys, or we're doomed.
        assert set(clusters.keys()) == set(centroids.keys())
        prev_centroids = deepcopy(centroids)
        
        ### STEP ONE -update clusters
        for x in X:
            #Check distances to all centroids
            bestind = -1
            bestdist = float('inf')
            for ind, centroid in centroids.iteritems():
                dist = _dist(x, centroid)
                if dist < bestdist:
                    bestdist = dist
                    bestind = ind
                #
            clusters[bestind].append(x)
        
        ### STEP TWO -update centroids
        for ind, points in clusters.iteritems():
            if not points:
                pass  #Cluster's empty - nothing to update
            else:
                centroids[ind] = np.mean(points, axis = 0)
        
        ### We're converged when all old centroids = new centroids.
        converged = all([_dist(prev_centroids[k],centroids[k]) == 0 
                    for k in xrange(K)])
        iterations += 1
        #
    return {tuple(centroids[i]) : clusters[i] for i in xrange(K)}
    
def lloyds(X, K, runs = 1, max_iterations = float('inf'), init = 'sample'):
    '''Runs Lloyd's algorithm to identify K clusters in the dataset X.
    X is a list of points like [[x1,y1],[x2,y2]---].
    Returns a hash of centroids mapping to points in the corresponding cluster.
    The objective is to minimize the sum of distances from each centroid to
    the points in the corresponding cluster. It might only converge on a local
    minimum, so the configuration with the lowest score (sum of distances) is
    returned.
    init denotes initialization mode, which can be 'sample', using a randomly
    select subset of the input data, or 'scatter', using random points selected
    from the same range as the data as initial centroids.
    
    Parameters
    ----------------
    X : array_like
      list of points. 2D example: [[3,4],[3.4, 7.2], ...]
    
    K : int
      Number of centroids
    
    runs : int
      Number of times to run the entire algorithm. The result with the lowest
      score will be returned.
    
    max_iterations : int or float
      Number of steps to allow each run. Default if infinit, i.e. the algorithm
      runs until it's fully converged.
    
    init : str
      Initialization mode. 'sample' means use a random subset of the data as
      starting centroids. 'scatter' means place starting centroids randomly in
      the entire x-y range of the dataset.
    
    Returns
    --------------
    result : dict
      A dictionary in which each key is a tuple of coordinated corresponding to
      a centroid, and each value is a list of points belonging to that cluster.
      '''
    
    record = float('inf')
    result = None
    for _ in xrange(runs):
        clusters = _lloyds_single_run(X, K, max_iterations = max_iterations,
                                      init = init)
        #Determine how good the clusters came out
        score = 0
        for centroid, points in clusters.iteritems():
            score += sum([_dist(centroid, p) for p in points or [] ])
        if score < record:
            result = clusters
            record = score
        #
    return result


def _makecolor():
    i = 0
    cols = ['b', 'g', 'r', 'c', 'm', 'y']
    while True:
        yield cols[i]
        i = (i+1)%len(cols)


def draw_clusters(clusters, threshold = 0, show = True, filename = None):
    '''Accepts a dict mapping cluster centroids to cluster points and makes
    a color-coded plot of them. Clusters containing fewer points than the
    threshold are plottet in black.'''
    colors = _makecolor()
    plt.figure()
    for centroid, points in clusters.iteritems():
        if not points:
            continue
        if len(points) < threshold:
            style = ['k,']
        else:
            color = colors.next()
            style = [color+'+']
            #Plot centroids
            x,y = centroid
            plt.plot(x,y, color = color, marker = 'd', markersize = 12)
        #plot points
        plt.plot(*(zip(*points)+style))
    if filename:
        plt.savefig(filename, bbox_inches = 'tight')
    if show:
        plt.show()


if __name__ == '__main__':
    points = [[random.uniform(-10,10), random.uniform(-10,10)] for _ in xrange(10**3)]
    clusters = lloyds(X = points, K = 6, runs = 1)
    draw_clusters(clusters = clusters, filename = 'lloyds_example.pdf')