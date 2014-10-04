'''
These classes permit to run the iterator in a another process and pass the data via pickling (slow, works always)
or shared memory (fast, works only for numeric numpy arrays of a defined and known in advance maximum size).

They can be used via multiple inheritance: the final dataset class simply inherits from both the 
SimpleDatasetScaffolding and MutiprocessingDatasetMixin.

Created on Aug 22, 2014

@author: chorows
'''

import numpy as np

import multiprocessing
from . import shmarray

from itertools import izip
import functools

from .. import SimpleDatasetScaffolding
from ..iterators import AbstractWrappedIterator, AbstractDataIterator
from collections import OrderedDict

class MultiprocessingPicklingDataIterator(AbstractWrappedIterator):
    def __init__(self, iterator, qlen=2, **kwargs):
        super(MultiprocessingPicklingDataIterator, self).__init__(iterator=iterator, **kwargs)
        assert qlen>1
        self.q = multiprocessing.Queue(qlen)
        self.proc = multiprocessing.Process(target=MultiprocessingPicklingDataIterator.__subprocess,
                                            args=(self.iterator, self.q))
        self.proc.start()
        
    @staticmethod
    def __subprocess(iterator, q):
        for batch in iterator:
            q.put(tuple(batch))
        q.put(StopIteration)
        
    def next(self):
        if self.proc is None:
            raise StopIteration
        o = self.q.get()
        if o is StopIteration:
            #is this needed?
            self.proc.terminate()
            self.proc.join()
            self.proc=None
            raise StopIteration
        return self.make(o)

    def close(self):
        if self.proc:
            self.proc.terminate()
            #while self.q.get(0)
            #    pass
            self.proc.join()
            self.proc=None
    
    def __del__(self):
        self.close()

class MultiprocessingSHMDataIterator(AbstractWrappedIterator):
    def __init__(self, iterator, qlen=2):
        assert qlen>1
        super(MultiprocessingSHMDataIterator, self).__init__(iterator=iterator)
        
        srcs = self.sources
        assert isinstance(srcs, OrderedDict)
        
        self.pipe, self.child_pipe = multiprocessing.Pipe()
        self.arrays = []
        for i in xrange(qlen+1):
            arr_tuple = tuple(shmarray.create(shape=(self.batch_size * np.prod(spc.dim), ), 
                                              dtype=spc.dtype) 
                              for spc in srcs.values()
                              ) 
            self.arrays.append(arr_tuple)
        
        for i in xrange(qlen):
            self.pipe.send(i)
        self.last_idx = qlen
        self.subproc = multiprocessing.Process(target=MultiprocessingSHMDataIterator.__subprocess,
                                               args=(self.iterator, self.child_pipe, self.arrays))
        self.subproc.start()
        
    @staticmethod
    def __subprocess(iterator, pipe, arrays):
        for batch in iterator:
            arr_idx = pipe.recv()
            shapes = tuple(a.shape for a in batch)
            #print 'got batch: ', batch,  ' shapes: ', shapes
            for da,ba in izip(arrays[arr_idx], batch):
                flat_ary = ba.ravel()
                #print 'da: ', da, 'flat ary; ', flat_ary
                da[:flat_ary.shape[0]] = flat_ary
            pipe.send((arr_idx, shapes))
        pipe.send(StopIteration)
        
    def next(self):
        if not self.subproc:
            raise StopIteration
        
        self.pipe.send(self.last_idx)
        o = self.pipe.recv()
        if o is StopIteration:
            self.subproc.join()
            self.subproc=None
            raise StopIteration
        arr_idx, shapes = o
        ret = self.make(a[:np.prod(s)].reshape(s) for a,s in izip(self.arrays[arr_idx],shapes))
        self.last_idx = arr_idx
        return ret
    
    def close(self):
        if self.subproc:
            self.subproc.terminate()
            #while self.q.get(0)
            #    pass
            self.subproc.join()
            self.subproc=None
    
    def __del__(self):
        self.close()

class MutiprocessingDatasetMixin(object):
    '''
    A mixin that fetchets the data in another process.
    
    Parameters
    ----------
    
    mp_method: one of pickle, shm
    '''    
    def __init__(self, *args, **kwargs):
        self._mp_method = kwargs.pop('mp_method','shm')
        self._qlen = kwargs.pop('qlen', 2)
        
        super(MutiprocessingDatasetMixin,self).__init__(*args, **kwargs)
        
    @functools.wraps(SimpleDatasetScaffolding._get_iterator)
    def _get_iterator(self, mode, batch_size, desired_sources, rng):
        if rng is not None:
            rng = np.random.RandomState(rng.randint(-9223372036854775808, 9223372036854775807, (10,)))
            
        gi_func = getattr(self, '_get_iterator_' + self._mp_method)
        return gi_func(mode, batch_size, desired_sources, rng)
    
    def _get_iterator_pickle(self, mode, batch_size, desired_sources, rng): 
        it = super(MutiprocessingDatasetMixin,self)._get_iterator(mode, batch_size, desired_sources, rng)
        return MultiprocessingPicklingDataIterator(it, qlen=self._qlen)

    def _get_iterator_shm(self, mode, batch_size, desired_sources, rng):
        it = super(MutiprocessingDatasetMixin,self)._get_iterator(mode, batch_size, desired_sources, rng)
        return MultiprocessingSHMDataIterator(it, qlen=self._qlen)
