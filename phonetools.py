# -*- coding: utf-8 -*-
"""
Created on Tue Dec 02 15:12:29 2014

@author: Bjarke
"""

import datetime
from pkg_resources import resource_filename
from ast import literal_eval as LE
import numpy as np
import json
import os

def _global_path(path):
    '''Helper method to ensure that data files belonging to the social_fabric
    module are available both when importing the module and when running
    individual parts from it for testing.
    usage: Always use _global_path(somefile) rather than somefile'''
    if __name__ == '__main__':
        return path
    else:
        return resource_filename('social_fabric', path)

def make_filename(prefix, interval, bin_size, ext=''):
    '''Returns a filename like "call_out-intervalsize-binsize_N", where N is
    an int so files aren't accidentally overwritten.
    I've been known to do that.'''
    stem = "%s-int%s-bin%s" % (prefix, interval, bin_size)
    n = 0
    ext = '.'+ext
    attempt = stem+ext
    while os.path.isfile(attempt):
        n+=1
        attempt = stem+"_"+str(n)+ext
    return attempt

def unix2str(unixtime):
    '''Converts timestamp to datetime object'''
    dt = datetime.datetime.fromtimestamp(unixtime)
    return str(dt)
    
#Converts bluetooth MAC-addresses to users
with open(_global_path('user_mappings/bt_to_user.txt'), 'r') as f:
    bt2user = LE(f.read())

#Converts phone 'number' code to users
with open(_global_path('user_mappings/phonenumbers.txt'), 'r') as f:
    number2user = LE(f.read())

#Converts IDs from psychological profile to users
with open(_global_path('user_mappings/user_mapping.txt'), 'r') as f:
    psych2user = {}
    #This is in tab-separated values, for some reason.
    for line in f.read().splitlines():
        (ID, usercode) = line.split('\t')
        assert ID.startswith('user_')
        psych2user[ID] = usercode

#Converts users to info on their psych profiles
with open(_global_path('user_mappings/user2profile.json'), 'r') as f:
    user2profile = json.load(f)

def is_valid_call(call_dict):
    '''Determine whether an entry is 'valid', i.e. make sure user isn't 
    calling/texting themselves, which people apparently do...'''
    caller = call_dict['user']
    try:
        receiver = number2user[call_dict['number']]
    except KeyError:
        receiver = None
    return caller != receiver

def readncheck(path):
    '''Reads in all valid call info from the file at path'''
    try:
        with open(path,'r') as f:
            raw = [LE(line) for line in f.readlines()]
    except IOError:
        return []  #no file :(
    #File read. Return proper calls
    return [call for call in raw if is_valid_call(call)]
        
class Binarray(list):
    '''Custom array type to automatically bin the time around a set center
    and place elements in each bin.
    The array can be centered using Binarray.center = <some time>.
    After centering, timestamps can be placed in bins around the center with
    Binarray.place_event(<some other time>).'''
    
    def __init__(self, interval = 3*60**2, bin_size = 3*60, center = None,
                 initial_values = None):
        '''
        Args:
        ----------------------
        interval : int
          Total number of seconds covered by the Binarray.
        
        bin_size : int
          Width of each bin measured in seconds. The total interval must be
          an integer multiplum of the bin size.
        
        center : int
          Where to center the Binarray. If the array is centered at time t,
          any event placed in it will placed in a bin depending on how long
          before or after t the event occured.
        
        initial_values : list
          List of values to start the Binarray with. Default is zeroes.'''
        
        
        #Make sure interval is an integer multiplum of bins
        if not interval % bin_size == 0:
            suggest = interval - interval % bin_size
            error = "Interval isn't an integer multiple of bin size. \
            Consider changing interval to %s." % suggest
            raise ValueError(error)
            
        #Set parameters
        self.bin_size = bin_size
        self.interval = interval
        self.size = 2*int(interval/bin_size)
        self.centerindex = int(self.size//2)
        self.center = center
        #Keep track of how many events missed the bins completely
        self.misses = 0
        #Call parent constructor
        if not initial_values:
            startlist = [0]*self.size
        else:
            if not len(initial_values) == self.size:
                msg = '''Array of start value must have length %d. Tried to
                instantiate with length of %d.''' % (self.size, 
                                                     len(initial_values))
                raise ValueError(msg)
            startlist = initial_values
        super(Binarray, self).__init__(startlist)
                    
    
    def place_event(self, position):
        '''Places one count in the appropriate bin if event falls within
        <interval> of <center>. Returns True on success.'''
        if self.center == None:
            raise TypeError('Center must be set!')
        delta = position - self.center
        #Check if event is outside current interval
        if np.abs(delta) >= self.interval:
            self.misses += 1
            return False
        #Woo, we're in the correct interval
        index = int(delta//self.bin_size)  #relative to middle of array
        self[self.centerindex + index] += 1
        return True
        
    def normalized(self):
        '''Returns a normalized copy of the array's contents.'''
        events = sum(self) + self.misses
        #Use numpy vectorized function for increased speed.
        f = np.vectorize(lambda x: x*1.0/events if events else 0)
        return f(self)
    
    def _todict(self):
        '''Helper method to allow dumping to JSON format.'''
        attrs = ['misses', 'interval', 'bin_size']
        d = {att : self.__getattribute__(att) for att in attrs}
        d['values'] = list(self)
        d['type'] = 'binarray'
        return d
        

def _dumphelper(obj):
    '''Evil recursive helper method to convert various nested objects to
    a JSON-serializeable format.
    This should only be called by the dump method!'''
    if isinstance(obj, Binarray):
        d = obj._todict()
        return  _dumphelper(d)
    elif isinstance(obj, tuple):
        hurrayimhelping = [_dumphelper(elem) for elem in obj]
        return {'type' : 'tuple', 'values' : hurrayimhelping}
    elif isinstance(obj, dict):
        temp = {'type' : 'dict'}
        contents = [{'key' : _dumphelper(key), 'value' : _dumphelper(value)}
                    for key, value in obj.iteritems()]
        temp['contents'] = contents
        return temp
    #Do nothing if obj is an unrecognized type. Let JSON raise errors.
    else:
        return obj

def _hook(obj):
    '''Evil recursive object hook method to reconstruct various nested
    objects from a JSON dump.
    This should only be called by the load method!'''
    if isinstance(obj, (unicode, str)):
        try:
            return _hook(LE(obj))
        except ValueError:  #happens for simple strings that don't need eval
            return obj
    elif isinstance(obj, dict):
        if not 'type' in obj:
            raise KeyError('Missing type info')
        if obj['type'] == 'dict':
            contents = obj['contents']
            d = {_hook(e['key']) : _hook(e['value']) for e in contents}
            #Make sure we also catch nested expressions
            if 'type' in d:
                return _hook(d)
            else:
                return d
        elif obj['type'] == 'binarray':
            instance = Binarray(initial_values = obj['values'],
                                bin_size = obj['bin_size'],
                                interval = obj['interval'])
            instance.misses = obj['misses']
            for key, val in obj.iteritems():
                if key == 'values':
                    continue
                instance.__setattr__(key, val)
            return instance
            
        elif obj['type'] == 'tuple':
            #Hook elements individually, then convert back to tuple
            restored = [_hook(elem) for elem in obj['values']]
            return tuple(restored)
        else:
            temp = {}
            for k, v in obj.iteritems():
                k = _hook(k)
                temp[k] = _hook(v)
            return temp
        #
    #Do nothing if obj is an unrecognized type
    else:
        return obj
    
def load(file_handle):
    '''Reads in json serialized nested combinations of dicts, binarrays
    and tuples.'''
    temp = json.load(file_handle, encoding='utf-8')
    return _hook(temp)

def dump(obj, file_handle):
    '''json serializes nested combinations of dicts, binarrays
    and tuples.'''
    json.dump(_dumphelper(obj), file_handle, indent=4, encoding='utf-8')


if __name__=='__main__':
    from time import time
    from random import randint
    #Create Binarray with interval +/- one hour and bin size ten minutes.
    ba = Binarray(interval = 60*60, bin_size = 10*60)
    #Center it on the present
    now = int(time())
    ba.center = now
    #Generate some timestamps around the present
    new_times = [now + randint(-60*60, 60*60) for _ in xrange(100)]
    for tt in new_times:
        ba.place_event(tt)
    
    #Save it
    with open('filename.sig', 'w') as f:
        dump(ba, f)
    
    print ba
