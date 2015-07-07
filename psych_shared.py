# -*- coding: utf-8 -*-
"""This module aims to allow sharing of some common methods and settings
when testing and tweaking various machine learning schemes.
Always import settings and the like from here!"""

from __future__ import division
import abc
from collections import Counter
import itertools
import json
import math
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import multiprocessing
import numpy as np
import random
from scipy.sparse import dok_matrix
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cross_validation import (cross_val_score, LeaveOneOut, KFold,
                                      StratifiedKFold)
import sys
from time import time
import traceback

oldhat = (35/256,39/256,135/256)
nude = (203/256,150/256,93/256)
wine = (110/256,14/256,14/256)
moerkeroed = (156/256,30/256,36/256)

def _make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

color_map = _make_colormap([oldhat, moerkeroed, 0.33, moerkeroed, nude, 0.67, nude])

big_five = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 
            'neuroticism']

#_default_features = ["n_texts",
#                     "ct_iet_std",
#                     "call_cir",
#                     "call_entropy",
#                     "text_cir",
#                     "n_calls",
#                     "text_latency",
#                     "call_outgoing",
#                     "fraction_social_time",
#                     "text_outgoing",
#                     "call_iet_std",
#                     "n_text_contacts",
#                     "call_night_activity",
#                     "call_iet_med",
#                     "outgoing_activity_AR_coeff_2",
#                     "text_entropy",
#                     "ct_cir",
#                     "text_response_rate",
#                     "n_ct_contacts",
#                     "social_entropy",
#                     "n_call_contacts",
#                     "n_ct",
#                     "text_iet_std",
#                     "ct_iet_med",
#                     "ct_entropy",
#                     "text_iet_med",
#                     "call_response_rate",
#                     "number_of_facebook_friends"]

_default_features = ['call_iet_med', 'text_iet_med', 'social_entropy',
'call_entropy', 'travel_med', 'n_places', 'text_latency',
'call_night_activity']

def split_ntiles(values, n):
    '''Determines the values that separate the imput list into n equal parts.
    this is a generalization of the notion of median (in the case n = 2) or
    quartiles (n=4).
    Usage: ntiles([5,6,7], 2) gives [6] for instance.'''
    result = []
    for i in xrange(1,n):
        percentile = 100/n * i
        result.append(np.percentile(values, percentile,
                                    interpolation='linear'))
    return result

def determine_ntile(value, ntiles):
    '''Determines which n-tile the input value belongs to.
    Usage: determine_ntile([7,9,13], 10) gives 2 (third quartile).
    This uses zero indexing so data split into e.g. quartiles will give results
    like 0,1,2,3 - NOT 1,2,3,4.'''
    #Check if value is outside either extreme, meaning n-tile 1 or n.
    if value >= ntiles[-1]:
        return len(ntiles)  #Remember the length is n-1
    elif value < ntiles[0]:
        return 0  #Values was in the first n-tile
    # Define possible region and search for where value is between two elements
    left = 0
    right = len(ntiles)-2
    #Keep checking th middle of the region and updating region
    ind = (right + left)//2
    while not ntiles[ind] <= value < ntiles[ind + 1]:
        #Check if lower bound tile is on the left
        if value < ntiles[ind]:
            right = ind - 1
        else:
            left = ind + 1
        ind = (right + left)//2
    # Being between ntiles 0 and 1, means second n-tile and so on.
    return ind + 1

def assign_labels(Y, n):
    '''Accepts a list and an int n and returns a list of discrete labels
    corresponding to the ntile each original y-value was in.'''
    ntiles = split_ntiles(Y, n)
    labels = [determine_ntile(y, ntiles) for y in Y]
    return labels

def normalize_data(list_):
    '''Normalizes input data to the range [-1, 1]'''
    lo, hi = min(list_), max(list_)
    if lo == hi:
        z = len(list_)*[0]
        return z
    else:
        return [2*(val-lo)/(hi - lo) - 1 for val in list_]


def read_data(filename, trait, n_classes = None, normalize = True,
               features='default', interpolate = True):
    '''This reads in a preprocessed datafile, splits psych profile data into 
    n classes if specified, filters desired psychological traits and
    features and returns as a tuple (X,Y, indexdict), which can be fed to a'
    number of off-the-shelf ML schemes.
    If trait=='Sex', female and male are converted to 0 and 1, respectively.
    indexdict maps each element of the feature vectors to their label, as in
    {42 : 'distance_travelled_pr_day'} etc.
    
    Args: 
      filename : str
        Name of the file containing the data.
      trait : str
        The psychological trait to extract data on.
      n_classes : int
        Number of classes to split data into. Default is None,
        i.e. just keep the decimal values. Ignored if trait == 'Sex', as data
        only has two discreet values.
      normalize : bool
        Whether to hard normalize data to [-1, 1].
      features : str/list
        Which features to read in. Can also be 'all'
        or 'default', meaning the ones I've pragmatically found to be 
        reasonable.

      interpolate : bool:
        Whether to replace NaN's with the median value of
        the feature in question.'''
    if trait == 'sex':    
        n_classes = None
    #Read in the raw data
    with open(filename, 'r') as f:
        raw = [json.loads(line) for line in f.readlines()]
    #Get list of features to be included - everything if nothing's specified
    included_features = []
    if features == 'default':
        included_features = _default_features
    elif features == 'all':
        included_features = raw[0]['data'].keys() if features=='all' else features
    else:
        included_features = features
    
    #Remove any features that only have NaN values.
    for i in xrange(len(included_features)-1,-1,-1):
        feat = included_features[i]
        if all(math.isnan(line['data'][feat]) for line in raw):
            del included_features[i]


    # ----- Handle feature vectors -----
    # Dict mapping indices to features
    indexdict = {ind : feat for ind, feat in enumerate(included_features)}
    # N_users x N_features array to hold data
    rows = len(raw)
    cols = len(included_features)
    X = np.ndarray(shape = (rows, cols))
    for i in xrange(rows):
        line = raw[i]
        #Construct data matrix
        for j, feat in indexdict.iteritems():
            val = line['data'][feat]
            X[i, j] = val
        #
    #Replace NaNs with median values
    if interpolate:
        for j in xrange(cols):
            #Get median of feature j
            med = np.median([v for v in X[:,j] if not math.isnan(v)])
            if math.isnan(med):
                raise ValueError('''Feature %s contains only NaN's and should
                                 have been removed.''' % indexdict[j])
            for i in xrange(rows):
                if math.isnan(X[i,j]):
                    X[i,j] = med

    if normalize:
        for j in xrange(cols):
            col = X[:,j]
            X[:,j] = normalize_data(col)        
    
    # ----- Handle class info -----
    trait_values = []
    for line in raw:
        #Add value of psychological trait
        psych_trait = line['profile'][trait]
        if trait == 'Sex':
            if psych_trait == 'Female':
                psych_trait = 0
            elif psych_trait == 'Male':
                psych_trait = 1
            else:
                raise ValueError('My code is binary gender normative, sorry.')
        trait_values.append(psych_trait)
    Y = []
    if n_classes == None:
        Y = trait_values
    else:
        ntiles = split_ntiles(trait_values, n_classes)
        Y = [determine_ntile(tr, ntiles) for tr in trait_values]
    
    return (X, Y, indexdict)


def plot_stuff(input_filename, output_filename=None, color=moerkeroed):
    with open(input_filename, 'r') as f:
        d = json.load(f)
    x = d['x']
    y = d['y']
    yerr = d['mean_stds']
    plt.plot(x,y, color=color, linestyle='dashed',
             marker='o')
    plt.errorbar(x, y, yerr=yerr, linestyle="None", marker="None",
                 color=color)
    if output_filename:
        plt.savefig(output_filename)

def get_TPRs_and_FPRs(X, Y, forest = None, verbose = False):
    '''Accepts a list of feature vectors and a list of labels and returns a
    tuple of true positive and false positive rates (TPRs and FPRs,
    respectively) for various confidence thresholds.'''
    kf = LeaveOneOut(n=len(Y))
    
    results = []
    thresholds = []
    
    counter = 0
    for train, test in kf:
        counter += 1
        if counter % 10 == 0 and verbose:
            print "Testing on user %s of %s..." % (counter, len(Y))
            
        result = {}
        train_data = [X[i] for i in train]
        train_labels = [Y[i] for i in train]
        test_data = [X[i] for i in test]
        test_labels = [Y[i] for i in test]
        
        if not forest:
            forest = RandomForestClassifier(
                                            n_estimators = 1000,
                                            n_jobs=-1,
                                            criterion='entropy')    
        
        forest.fit(train_data, train_labels)
        result['prediction'] = forest.predict(test_data)[0]
        result['true'] = test_labels[0]
        confidences = forest.predict_proba(test_data)[0]
        result['confidences'] = confidences
        thresholds.append(max(confidences))
        
        results.append(result)
    
    #ROC curve stuff - false and true positive rates
    TPRs = []
    FPRs = []
    
    unique_thresholds = sorted(list(set(thresholds)), reverse=True)
    
    for threshold in unique_thresholds:
        tn = 0
        fn = 0
        tp = 0
        fp = 0
        for result in results:
            temp = result['prediction']
            if temp == 1 and result['confidences'][1] >= threshold:
                pred = 1
            else:
                pred = 0
            if pred == 1:
                if result['true'] == 1:
                    tp += 1
                else:
                    fp += 1
                #
            elif pred == 0:
                if result['true'] == 0:
                    tn += 1
                else:
                    fn += 1
                #
            #
        TPRs.append(tp/(tp + fn))
        FPRs.append(fp/(fp + tn))
    return (TPRs, FPRs)

def make_roc_curve(TPRs, FPRs, output_filename = None):
    '''Accepts a list of true and false positive rates (TPRs and FPRs,
    respectively) and generates a ROC-curve.'''
    predcol = moerkeroed
    basecol = oldhat
    fillcol = nude
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    TPRs = [0] + TPRs + [1]
    FPRs = [0] + FPRs + [1]
    
    area = 0.0
    for i in xrange(len(TPRs)-1):
        dx = FPRs[i+1] - FPRs[i]
        y = 0.5*(TPRs[i] + TPRs[i+1])
        under_curve = dx*y
        baseline = dx*0.5*(FPRs[i] + FPRs[i+1])
        area += under_curve - baseline
    
    baseline = FPRs
    ax.fill_between(x = FPRs, y1 = TPRs, y2 = baseline, color = fillcol,
                     interpolate = True, alpha=0.8)
    ax.plot(baseline, baseline, color = basecol, linestyle = 'dashed',
             linewidth = 1.0, label = 'Baseline')
    ax.plot(FPRs, TPRs, color=predcol, linewidth = 1,
                       label = 'Prediction')
                       
    plt.xlabel('False positive rate.')
    plt.ylabel('True positive rate.')
    
    handles, labels = ax.get_legend_handles_labels()
    hest = mpatches.Patch(color=fillcol)

    labels += ['Area = %.3f' % area]
    handles += [hest]
    ax.legend(handles, labels, loc = 'lower right')
#    plt.legend(handles = [tp_line, base])
    if output_filename:
        plt.savefig(output_filename)
    plt.show()

def rank_features(X, Y, forest, indexdict, limit = None):
    '''Ranks the features of a given dataset and classifier.
    indexdict should be a map from indices to feature names like
    {0 : 'average_weigth'} etc.
    if limit is specified, this method returns only the top n ranking features.
    Returns a dict like {'feature name' : (mean importances, std)}.'''
    importances = forest.feature_importances_
    stds=np.std([tree.feature_importances_ for tree in forest.estimators_],
                axis=0)
    indices = np.argsort(importances)[::-1]
    if limit:
        indices = indices[:limit]
    d = {indexdict[i] : (importances[i], stds[i]) for i in indices}
    return d

def check_performance(X, Y, clf, strata = None):
    '''Checks forest performance compared to baseline.'''
    N_samples = len(Y)
    #Set up validation indices
    if not strata:  #Do leave-one-out validation
        skf = KFold(N_samples, n_folds=N_samples, shuffle = False)
    else:  #Do stratified K-fold
        skf = StratifiedKFold(Y, n_folds=strata)
        
    #Evaluate classifier performance
    scores = []
    for train, test in skf:
        train_data = [X[ind] for ind in train]
        train_labels = [Y[ind] for ind in train]
        test_data = [X[ind] for ind in test]
        test_labels = [Y[ind] for ind in test]
        
        #Check performance of input forest
        clf.fit(train_data, train_labels)
        score = clf.score(test_data, test_labels)
        scores.append(score)
    
    #Compute baseline
    most_common_label = max(Counter(Y).values())
    baseline = float(most_common_label)/N_samples
    
    #Compare results with prediction baseline
    score_mean = np.mean(scores)/baseline
    score_std = np.std(scores)/baseline
    
    return (score_mean, score_std)

def check_regressor(X, Y, reg, strata = None):
    '''Checks the performance of a regressor against mean value baseline.'''
    N_samples = len(Y)
    #Set up validation indices
    if not strata:  #Do leave-one-out validation
        skf = KFold(N_samples, n_folds=N_samples,
                                     shuffle = False)
    else:  #Do stratified K-fold
        skf = StratifiedKFold(Y, n_folds=strata)
        
    #Evaluate performance
    model_abs_errors = []
    baseline_abs_errors = []
    for train, test in skf:
        train_data = [X[ind] for ind in train]
        train_labels = [Y[ind] for ind in train]
        test_data = [X[ind] for ind in test]
        test_labels = [Y[ind] for ind in test]
        
        #Check performance of input forest
        reg.fit(train_data, train_labels)
        base = np.mean(train_labels)
        for i in xrange(len(test_data)):
            pred = reg.predict(test_data[i])
            true = test_labels[i]
            model_abs_errors.append(np.abs(pred - true))
            baseline_abs_errors.append(np.abs(base - true))
        #
    return (np.mean(model_abs_errors), np.mean(baseline_abs_errors))

class _RFNN(object):
    __metaclass__ = abc.ABCMeta
    '''Abstract class for random forest nearest neightbor predictors.
    This should never be instantiated.'''
    
    def __init__(self, forest, n_neighbors):
        self.forest = forest
        self.n_neighbors = n_neighbors
        self.X = None
        self.Y = None
    
    def fit(self, X, Y):
        '''Fits model to training data.
        
        Args
        ------------
        X : List
          List of training feature vectors.
        
        Y : List
          List of training labels or values to be predicted.'''
        if not len(X) == len(Y):
            raise ValueError("Training input and output lists must have "
                             "same length.")
        if not self.n_neighbors <= len(X):
            raise ValueError("Fewer data points than neighbors.")
            
        self.forest.fit(X, Y)
        self.X = X
        self.Y = Y

    def _rf_similarity(self, a, b):
        '''Computes a similarity measure for two points using a trained random
        forest classifier.'''
        if self.X == None or self.Y == None:
            raise NotImplementedError("Model has not been fittet to data yet.")

        #Feature vectors must be single precision.
        a = np.array([a], dtype = np.float32)
        b = np.array([b], dtype = np.float32)
        hits = 0
        tries = 0
        for estimator in self.forest.estimators_:
            tries += 1
            tree = estimator.tree_
            # Check whether the points end up on the same leaf for this tree
            if tree.apply(a) == tree.apply(b):
                hits += 1
            #
        return hits/tries

    def find_neighbors(self, point):
        '''Determine the n nearest nieghbors for the given point.
        Returns a list of n tuples like (yval, similarity).
        The tuples are sorted descending by similarity.'''
        if self.X == None or self.Y == None:
            raise NotImplementedError("Model has not been fittet to data yet.")

        #Get list of tuples like (y, similarity) for the n 'nearest' points
        nearest = [(None,float('-infinity')) for _ in xrange(self.n_neighbors)]
        for i in xrange(len(self.X)):
            similarity = self._rf_similarity(self.X[i], point)
            # update top n list if more similar than the furthest neighbor
            if similarity > nearest[-1][1]:
                nearest.append((self.Y[i], similarity))
                nearest.sort(key = lambda x: x[1], reverse = True)
                del nearest[-1]
            #
        return nearest
    
    #Mandatory methods - must be overridden
    @abc.abstractmethod
    def predict(self, point):
        pass
    
    @abc.abstractmethod
    def score(self, X, Y):
        pass

def _reservoir_sampler(start = 1):
    '''Generator of the probabilities need to do reservoir sampling. The point
    it that this can be used to iterate through a list, discarding each element
    for the following element with probability P_n and ending up with a random 
    element from the list.'''
    n = start
    while True:
        p = 1/n
        r = random.uniform(0,1)
        if r < p:
            yield True
        else:
            yield False
        n += 1

class RFNNClassifier(_RFNN):
    '''Random Forest Nearest Neighbor Classifier.
    
    Parameters
    ----
    n_neighbors : int
      Number of neighbors to consider.
    
    forest : RandomForestClassifier
      The forest which will provide a 
      distance measure on which determine nearest neighbors.
    
    weighting : str
      How to weigh the votes of different neighbors.
      'equal' means each neighbor has an equivalent vote.
      'linear' mean votes are weighed by their similarity to the input point.
    '''
    def predict(self, point):
        '''Predicts the label of a given point.'''
        neighbortuples = self.find_neighbors(point)
        if self.weighting == 'equal':
            #Simple majority vote. Select randomly if it's a tie.
            predictions = [t[0] for t in neighbortuples]
            best = 0
            winner = None
            switch = _reservoir_sampler(start = 2)
            for label, votes in Counter(predictions).iteritems():
                if votes > best:
                    best = votes
                    winner = label
                    switch = _reservoir_sampler(start = 2)
                elif votes == best:
                    if switch.next():
                        winner = label
                    else:
                        pass
                else:
                    pass
                #
            return winner
        
        #Weigh votes by their similarity to the input point
        elif self.weighting == 'linear':
            #The votes are weighted by their similarity
            d = {}
            for yval, similarity in neighbortuples:
                try:
                    d[yval] += similarity
                except KeyError:
                    d[yval] = similarity
            best = float('-infinity')
            winner = None
            for k, v in d.iteritems():
                if v > best:
                    best = v
                    winner = k
                else:
                    pass
            return winner
    
    def score(self, X, Y):
        if not len(X) == len(Y):
            raise ValueError("Training data and labels must have same length.")
        
        hits = 0
        n = len(X)
        for i in xrange(n):
            pred = self.predict(X[i])
            if pred == Y[i]:
                hits += 1
            #
        return hits/n
            
    
    def __init__(self, forest = None, n_neighbors = 3, weighting = 'equal',
                 n_jobs = 1):
        #Make sure we have a forest *classifier*
        if forest == None:
            forest = RandomForestClassifier(n_estimators = 1000,
                                            criterion = 'entropy',
                                            n_jobs = n_jobs)
        if not isinstance(forest, RandomForestClassifier):
            raise TypeError("Forest must be a classifier")
        
        self.weighting = weighting
        
        #Call parent constructor
        super(RFNNClassifier, self).__init__(forest, n_neighbors)

class RFNNRegressor(_RFNN):
    '''Random Forest Nearest Neighbor Regressor.
    
    Parameters
    ----
    n_neighbors : int
      Number of neighbors to consider.
    
    forest : RandomForestRegressor
      The forest which will provide a 
      distance measure on which determine nearest neighbors.
    
    weighting : str
      How to weigh the votes of different neighbors.
      'equal' means each neighbor has an equivalent weight.
      'linear' mean votes are weighed by their similarity to the input point.
    '''
    def predict(self, point):
        # lists of the y vaues and similarities of nearest neighbors
        neighbortuples = self.find_neighbors(point)
        yvals, similarities = zip(*neighbortuples)
        
        # Weigh each neighbor y value equally is that's how we roll
        if self.weighting == 'equal':
            weight = 1.0/len(yvals)
            result = 0.0
            for y in yvals:
                result += y*weight
            return result
            #
        # Otherwise, weigh neighbors by similarity
        elif self.weighting == 'linear':
            weight = 1.0/(sum(similarities))
            result = 0.0
            for i in xrange(len(yvals)):
                y = yvals[i]
                similarity = similarities[i]
                result += y*similarity*weight
            return result
        
    
    def score(self, X, Y):
        if not len(X) == len(Y):
            raise ValueError("X and Y must be same length.")
        errors = [Y[i] - self.predict(X[i]) for i in xrange(len(X))]
        return np.std(errors)
        
    
    def __init__(self, forest = None, n_neighbors = 3, weighting = 'equal'):
        #Check forest type.
        if forest == None:
            forest = RandomForestRegressor(n_estimators = 1000, n_jobs = -1)
        if not isinstance(forest, RandomForestRegressor):
            raise TypeError("Must use Random Forest Regressor to initialize.")
        # Set params
        self.weighting = weighting
        # Done. Call parent constructor
        super(RFNNRegressor, self).__init__(forest, n_neighbors)    


class _BaselineRegressor(object):
    '''Always predicts the mean of the training set.'''
    def __init__(self, guess=None):
        self.guess = guess
    def fit(self, xtrain, ytrain):
        '''Find the average of input lidt of target values and guess on that
        from now on.'''
        self.guess = np.mean(ytrain)
    def predict(self, x):
        return self.guess


class _BaselineClassifier(object):
    '''Always predicts the most common label in the training set'''
    def __init__(self, guess=None):
        self.guess = guess    
    def fit(self, xtrain, ytrain):
        '''Find the most common label and guess on that from now on.'''
        countmap = Counter(ytrain)
        best = 0
        for label, count in countmap.iteritems():
            if count > best:
                best = count
                self.guess = int(label)
            #
        #    
    def predict(self, x):
        return self.guess
    
    
    

def _worker(X, Y, score_type, train_percentage, classifier, clf_args, n_groups,
            replace):
    '''Worker method for parallelizing bootstrap evaluations.'''
    #Create bootstrap sample
    try:
        rand = np.random.RandomState()  #Ensures PRNG works in children
        indices = rand.choice(xrange(len(X)), size = len(X), replace = replace)
        rand.randint
        xsample = [X[i] for i in indices]
        ysample = [Y[i] for i in indices]
        #Create regressor if we're doing regression
        if classifier == 'RandomForestRegressor':
            clf = RandomForestRegressor(**clf_args)
        elif classifier == 'SVR':
            clf = svm.SVR(**clf_args)
        elif classifier == 'baseline_mean':
            clf = _BaselineRegressor()
        
        #Create classifier and split dataset into labels
        elif classifier == 'RandomForestClassifier':
            clf = RandomForestClassifier(**clf_args)
            ysample = assign_labels(ysample, n_groups)
        elif classifier == 'SVC':
            clf = svm.SVC(**clf_args)
            ysample = assign_labels(ysample, n_groups)
        elif classifier == 'baseline_most_common_label':
            clf = _BaselineClassifier()
            ysample = assign_labels(ysample, n_groups)
        #Fail if none of the above classifiers were specified
        else:
            raise ValueError('Regressor or classifier not defined.')
        #Generate training and testing set
        cut = int(train_percentage*len(X))
        xtrain = xsample[:cut]
        ytrain = ysample[:cut]
        xtest = xsample[cut:]
        ytest = ysample[cut:]
        
        #Fit the classifier or regressor
        clf.fit(xtrain, ytrain)
    
        #Compute score and append to output list
        if score_type == 'mse':
            scores = [(ytest[i] - clf.predict(xtest[i]))**2
                  for i in xrange(len(xtest))]
            return np.mean(scores)
    
        elif score_type == 'fraction_correct':
            n_correct = sum([ytest[i] == int(clf.predict(xtest[i]))
                            for i in xrange(len(ytest))])
            score = n_correct/len(ytest)
            return score
        elif score_type == 'over_baseline':
            #Get score
            score = sum([ytest[i] == int(clf.predict(xtest[i]))
                            for i in xrange(len(ytest))])
            #Get baseline
            baselineclf = _BaselineClassifier()
            baselineclf.fit(xtrain, ytrain)
            baseline = sum([ytest[i] == baselineclf.predict(xtest[i])
                            for i in xrange(len(ytest))])
            return score/baseline
        
        #Fail if none of the above performance metrics were specified
        else:
            raise ValueError('Score type not defined.')
        
        #Job's done!
        return None
    except:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def bootstrap(X, Y, classifier, score_type = 'mse', train_percentage = 0.8,
              clf_args = {}, iterations = 1000, n_groups = 3, n_jobs = 1,
              replace = True):
    '''Performs bootstrap resampling to evaluate the performance of some 
    classifier or regressor. Note that this takes the *complete dataset* as 
    arguments as well as arguments specifying which predictor to use and which
    function to estimate the distribution of.
    This seems to be the most straightforward generalizable implementation
    which can be parallelized, as passing e.g. the scoring function directly
    clashed with the mechanisms implemented to work around the GIL for 
    multiprocessing for obscure reasons.

    Parameters:
    ----------------
    X : list
      All feature vectors in the complete dataset.
      
    Y : list
      All 'true' labels or output values in the complete dataset.
    
    classifier : str
      Which classifier to use to predict the test set. Allowed values:
      'RandomForestRegressor', 'baseline_mean', 'SVR',
      'RandomForestClassifier', 'SVC', 'baseline_most_common_label'
    
    score_type : str
      String signifying which function to estimate the distribution of.
      Allowed values: 'mse', 'fraction_correct', 'over_baseline'
    
    train_percentage : float
      The percentage [0:1] of each bootstrap sample to be used for training.
    
    clf_args : dict
      optional arguments to the constructor method of the regressor/classifier.
      
    iterations : int
      Number of bootstrap samples to run.
    
    n_jobs : int
      How many cores (maximum) to use.
    
    replace : bool
      Whether to sample with replacement when obtaining the bootstrap samples.
    '''
    
    if not len(X) == len(Y):
        raise ValueError("X and Y must have equal length.")

    #Arguments to pass to worker processes
    d = {'X' : X, 'Y' : Y, 'train_percentage' : train_percentage,
         'classifier' : classifier, 'clf_args' : clf_args,
         'score_type' : score_type, 'n_groups' : n_groups,
         'replace' : replace}
    
    #Make job queue
    pool = multiprocessing.Pool(processes = n_jobs)
    jobs = [pool.apply_async(_worker, kwds = d) for _ in xrange(iterations)]
    pool.close()  #run
    pool.join()  #Wait for remaining jobs
    
    #Make sure no children died too early
    if not all(job.successful() for job in jobs):
        raise RuntimeError('Some jobs failed.')
    
    return [j.get() for j in jobs]

def get_correlations(X, Y):
    '''Given a list of feature vectors X and labels or values Y, returns a list
    of correlation coefficients for each dimension of the feature vectors.'''
    n_feats = len(X[0])
    correlations = []
    for i in xrange(n_feats):
        temp = np.corrcoef([x[i] for x in X], Y)
        correlation = temp[0,1]
        if math.isnan(correlation):
            correlation = 0
        correlations.append(correlation)
    return correlations
        

def make_kernel(importances = None, gamma = 1.0, threshold = 0.0):
    '''Returns a weighted radial basis function (WRBF) kernel which can be 
    passed to an SVM or SVR from the sklearn module.
    
    Parameters:
    -----------------------
    importances : list
      The importance of each input feature. The value of element i can mean
      e.g. the linear correlation between feature i and target variable y.
    
    gamma : float
      The usual gamma parameter denoting inverse width of the gaussian used.
      
    threshold : float
      The minimum value of the importance to be used. Features less important
      than threshold will simply be ignored.
    '''
    def kernel(x,y, *args, **kwargs):
        if importances == None:
            _corrs = np.ones(shape = (len(x),), dtype = np.float64)
        else:
            _corrs = importances
        d = len(_corrs)  #number of features
        #Strong (above threshold) correlations
        strong = [np.abs(c) if np.abs(c) >= threshold else 0.0
                  for c in _corrs]
        normfactor = 1.0/np.sqrt(sum([e**2 for e in strong]))
        #Metric to compute distance between points
        metric = dok_matrix((d,d), dtype = np.float64)
        for i in xrange(d):
            metric[i,i] = strong[i]*normfactor
        # 
        result = np.zeros(shape = (len(x), len(y)))
        for i in xrange(len(x)):
            for j in xrange(len(y)):
                dist = x[i] - y[j]
                result[i,j] = np.exp(-gamma*np.dot(dist,dist))
        return result
    return kernel

if __name__ == '__main__':
    pass
    X, Y, ind_dict = read_data('../data.json', trait = 'openness',
                        features = ['call_iet_med', 'text_iet_med', 'social_entropy', 'call_entropy', 'travel_med', 'n_places', 'text_latency', 'call_night_activity'],
                        n_classes = 3
                        )
#    X = [[1,2,7,0],[3,1,6,0.01],[6,8,1,0],[10,8,2,0.01]]
#    Y = [1,1,0,0]
    
    cut = int(0.8*len(X))
    
    xtrain = X[:cut]
    ytrain = Y[:cut]
    xtest = X[cut:]
    ytest = Y[cut:]
    
    corrs = get_correlations(X, Y)
    print corrs
    
    C = 70
    gamma = 3.75
    
    kernel = make_kernel(corrs, 0.05)
    
    clf = svm.SVC(kernel = kernel)
    clf.fit(xtrain, ytrain)
    
    hits = 0
    
    for i in xrange(len(xtest)):
        if clf.predict(xtest[i]) == ytest[i]:
            hits += 1
        #
    print 100.0*hits/len(ytest)
    
    for i in xrange(len(corrs)):
        print ind_dict[i], corrs[i]
    


#    print len(X[0])
##    print i
#    print [el for el in i.values() if 'init' in el]