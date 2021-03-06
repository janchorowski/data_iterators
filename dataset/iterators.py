'''
Created on Aug 21, 2014

@author: chorows
'''

import logging
logger = logging.getLogger(__file__)

from itertools import izip
from collections import OrderedDict, namedtuple

import numpy as np

from pylearn2.utils.rng import make_np_rng

class AbstractDataIterator(object):
    """
The AbstractDataIterator implements the iterator protocol with the following additions:
1. It returns namedtuples of numpy arrays only
2. The namedtuples are created by self.OutputClass
3. The names of sources it carries can be queried by self.source_names property. 
   They correspond to the fields of the namedtuple
4. It has the following properties that are needed by PyLearn2 trainers:
    - stochastic
    - num_examples
    - num_batches
    - batch_size
    - uneven
   The properties are read from the dictionary self.properties. Defaults are provided for all values.

Each element returned by the iterator is one batch of data. Individual examples in the batch are 
indexed by the first dimension, the interpretation of the other dimensions is left to the processing functions.
   
The idea is that the data preprocessing pipeline will be created by plugging those iterators, 
specific sources can be always queried by name.
    """
    
    @staticmethod
    def create_output_class(source_names):
        return namedtuple(
                    typename='OutTuple_'+'_'.join(source_names), 
                    field_names=source_names)
    
    def __init__(self, sources = None, source_names=None, OutputClass=None, properties=None, **kwargs):
        """Initialize the required fields: sources, source_names, OutputClass, and properies.
        
        Arguments:
        ----------
            Those arguments are mutually exclusive:
                sources : dict of name -> Space
                source_names : list of strings
                OutputClass : a namedtuple class
        
        Keyword Arguments:
        ------------------
            properties: dictionary of iterator's properties. Special values recognized are:
                stochastic, num_examples, num_batches, batch_size, uneven
            
            **kwargs: passthrough for multiple inheritance
        """
        super(AbstractDataIterator, self).__init__(**kwargs)
        if properties is None:
            properties = {}
        num_args = sum(i is not None for i in [sources, source_names, OutputClass])
        if num_args != 1:
            raise Exception("Must provide exactly one of sources, source_names, OutClass! (provided %d)" % (num_args, ))
        if OutputClass is not None:
            self.OutputClass = OutputClass
        else:
            if source_names is None:
                source_names=sources.keys()
                properties['sources'] = sources
            self.OutputClass = AbstractDataIterator.create_output_class(source_names)
                    
        self.properties=properties

    
    #Mandatory functions to override:
    def next(self):
        raise NotImplementedError()
    
    #Normal operation
    def make(self, iterable):
        """Make a namedtuple returned by this iterator.
        """
        return self.OutputClass._make(iterable)
    
    @property
    def source_names(self):
        return self.OutputClass._fields
    
    @property
    def sources(self):
        sources = self.properties['sources']
        ret = OrderedDict()
        for sn in self.source_names:
            ret[sn] = sources[sn]
        return ret
    
    @property
    def stochastic(self):
        return self.properties.get('stochastic', None)
    
    @property
    def num_examples(self):
        return self.properties.get('num_examples', float('nan'))
    
    @property
    def num_batches(self):
        return self.properties.get('num_batches', float('nan'))
    
    @property
    def batch_size(self):
        return self.properties.get('batch_size', float('nan'))
    
    @property
    def uneven(self):
        return self.properties.get('uneven', True)
    
    def __repr__(self):
        return "DataIterator returning %s" % (self.source_names, )
        
    def __iter__(self):
        return self
    

class AbstractWrappedIterator(AbstractDataIterator):
    """An abstract class for an iterator that wraps other iterators and transforms its data
    """
    def __init__(self, iterator, **kwargs):
        """
        Initialize the iterator transformer.
        
        Arguments:
        ----------
            iterator: wrapped iterator
            
        The constructor saves the iterator into self.iterator and copies its properties.
        """
        super(AbstractWrappedIterator, self).__init__(OutputClass = iterator.OutputClass, **kwargs)
        self.iterator = iterator.__iter__()
        self.properties = dict(self.iterator.properties)
        
        #unless a class restores it, we are better off assuming they have changed??
        #self.properties.sources=None  


class DataIterator(AbstractDataIterator):
    """A wrapper for Python Iterators that conforms to the interface of AbstractDataIterator
    """
    def __init__(self, iterator, source_names, **kwargs):
        """Wraps the python iterator and names the values it yields.
        
        Arguments:
        ----------
            iterator: python iterator to wrap. It must return tuples of nympy arrays
            source_names: names of elements returned by the iterator
        """
        super(DataIterator, self).__init__(source_names=source_names, **kwargs)
        self.iterator = iterator
    
    def next(self):
        return self.make(self.iterator.next())


class NamedTupleIterator(AbstractDataIterator):
    """An iterator that wraps a Python iterator that already returns named tuples.
    """
    def __init__(self, iterator, OutputClass, **kwargs):
        """Initialize the iterator by reusing the namedtuple class.
        
        Arguments:
        ----------
            iterator: the python iterator that yields namedtuples created using OutputClass constructor
            OutputClass: the constructor for returned namedtuples
        """
        super(NamedTupleIterator, self).__init__(OutputClass=OutputClass, **kwargs)
        self.iterator = iterator
    
    def next(self):
        return self.iterator.next()


class TransformingIterator(AbstractWrappedIterator):
    """An iterator that transforms the batches that pass through it. 
    """
    def __init__(self, iterator, transforms, drop_untransformed=False, **kwargs):
        """
        Arguments:
        ----------
            iterator: wrapped iterator
            transforms: a dictionary of source_names into functions to transform the data
            drop_untransformed: if True remove sources which are not transformed by this iterator
            
        Notes:
        ------
            since data transformation will probably invalidate the Space, we drop the sources property.
        """
        super(TransformingIterator, self).__init__(iterator=iterator, **kwargs)
        if drop_untransformed:
            self.OutputClass = AbstractDataIterator.create_output_class(transforms.keys())
        
        #these will get changed, the rest souldn't
        self.properties.pop('sources',None)
        
        self.transforms = transforms
        if type(iterator)==TransformingIterator:
            self.iterator = iterator.iterator
            self_trans = self.transforms
            self.transforms = {}
            other_trans = iterator.transforms
            for s in self.source_names:
                of = other_trans.get(s, None)
                sf = self_trans.get(s, None)
                if of is None:
                    if sf is not None:
                        self.transforms[s]=sf
                else:
                    if sf is None:
                        self.transforms[s]=of
                    else:
                        self.transforms[s] = lambda X, of=of, sf=sf: sf(of(X)) 
        
    def next(self):
        t = self.transforms
        utt = self.iterator.next()
        return self.make(t[s](getattr(utt, s)) if s in t else getattr(utt, s) for s in self.source_names)

class LimitBatchSizeIterator(AbstractWrappedIterator):
    """
    Iterator that splits large batches into smaller ones.
    """
    def __init__(self, iterator, batch_size):
        """
        Arguments:
        ----------
            iterator: the wrapped iterator
            batch_size: the maximum number of examples in the transfrmed batches
        """
        super(LimitBatchSizeIterator, self).__init__(iterator=iterator)
        self.properties['batch_size'] = batch_size
        #we don't know how many batches will be created so we just drop the property 
        self.properties.pop('num_batches', None)
        self.batch_queue = []
        
    def next(self):
        if not self.batch_queue:
            utt = self.iterator.next()
            cutpoints = range(self.batch_size, utt[0].shape[0], self.batch_size)
            utt_split = [np.split(u, cutpoints) for u in utt]
            m = self.OutputClass._make
            self.batch_queue = [m(u) for u in izip(*utt_split)]
            self.batch_queue.reverse()
        return self.batch_queue.pop()

class _BatchIt:
    "Helper c class to hold an batch along with the index of the currrent row"
    
    __slots__ = ['batch','pos','len']
    
    def __init__(self, batch, permutation):
        self.batch=batch
        self.pos=0
        self.len=batch[0].shape[0]
        self.permutation = permutation

class ShuffledExamplesIterator(AbstractWrappedIterator):
    """An iterator that reads a number of batches into a shuffling memory, then creates batches by randomly selecting examples
    from the pool.
    """
    _default_seed = (17, 2, 946)
    
    def __init__(self, iterator, batch_size, 
                 shuffling_mem=100e6, rng=_default_seed):
        """
        Arguments:
        ----------
            iterator: an iterator returning batches of examples
            batch_size: the maximum size of produced batches
            shuffling_mem: how much memory to approximately use for the shuffling pool
            rng: random number generator 
        """
        super(ShuffledExamplesIterator, self).__init__(iterator=iterator)
        self.shuffling_mem = shuffling_mem
        self.properties['batch_size'] = batch_size
        self.properties.pop('num_batches', None)
        self.properties['stochastic'] = True
        self.batch_pool = None
        self.mem_used = 0
        self.rng = make_np_rng(rng, which_method='random_integers')
        
    def _fill_to_mem_limit(self):
        for batch in self.iterator:
            self.mem_used += sum(u.nbytes for u in batch)
            permutation = self.rng.permutation(batch[0].shape[0]) 
            self.batch_pool.append(_BatchIt(batch, permutation))
            if self.mem_used >= self.shuffling_mem:
                break
        
    def next(self):
        #delay pool filling so that iterator creation is fast
        if self.batch_pool is None:
            self.batch_pool = []
            self._fill_to_mem_limit()
        batch_pool = self.batch_pool
        if not batch_pool:
            raise StopIteration
        
        #positions = self.rng.choice(len(batch_pool), self.batch_size)
        positions = self.rng.randint(len(batch_pool), size=self.batch_size)
        
        ret = self.make(np.empty((self.batch_size, batch.shape[1]), dtype=batch.dtype) 
                    for batch in batch_pool[0].batch)
        
        #fill row by row
        i=0
        while i < positions.shape[0]:
            p = positions[i]
            batch = batch_pool[p]
            if batch.pos<batch.len:
                batchelem = batch.permutation[batch.pos]
                for r,u in izip(ret,batch.batch):
                    r[i,...] = u[batchelem,...]
                batch.pos += 1
                i += 1
            else: #utt.pos >= utt.len
                # swap with last unless we are the last
                last = batch_pool.pop()
                if last!=batch:
                    batch_pool[p] = last
                self.mem_used -= sum(u.nbytes for u in batch.batch)
                
                #fetch new utterances
                self._fill_to_mem_limit()
                
                if len(batch_pool)==0:
                    #return what we have right now...
                    return self.make(r[:i,...] for r in ret)
                
                rem_pos = positions[i:] #take a view
                bad_idx = rem_pos >= len(batch_pool)
                #rem_pos[bad_idx] = self.rng.choice(len(batch_pool), bad_idx.sum())
                rem_pos[bad_idx] = self.rng.randint(len(batch_pool), size=bad_idx.sum())
        return ret

class DataSpaceConformingIterator(AbstractWrappedIterator):
    """An iterator that converts between different DataSpaces
    """
    def __init__(self, iterator, destination_data_specs, iterator_data_specs=None, rename_map=None, **kwargs):
        """
        Arguments:
        ----------
            iterator: wrapped iterator
            destination_data_specs: data specs for the returned batches
            iterator_data_specs: ovverride value for the data specs of the wrapped iterator. Must be set if the 
                iterator does not provide data specs through the sources property.
            rename_map: a dict mapping new source names into source names used by the wrapped iterator
        """
        super(DataSpaceConformingIterator, self).__init__(iterator=iterator, **kwargs)
        
        if iterator_data_specs is None:
            iterator_data_specs = self.iterator.sources
        
        if rename_map is not None:
            self.OutputClass = AbstractDataIterator.create_output_class(destination_data_specs.keys())
        else:
            rename_map = dict((s,s) for s in self.source_names)
        
        iterator_source_names = iterator.source_names
        
        self._source_idxs = dict((sn, iterator_source_names.index(rename_map[sn])) for sn in self.source_names)
        self._conversion_funcs = {}
        
        for sn in self.source_names:
            iterator_space = iterator_data_specs[rename_map[sn]]
            destination_space = destination_data_specs[sn]
            def cf(X, iter_sp=iterator_space, dest_sp=destination_space):
                return iter_sp.np_format_as(X, dest_sp)
            self._conversion_funcs[sn] = cf
        
        self.properties['sources'] = destination_data_specs
    
    def next(self):
        batch = self.iterator.next()
        cf = self._conversion_funcs
        src_idx = self._source_idxs
        ret = self.make(cf[sn](batch[src_idx[sn]]) for sn in self.source_names)
        return ret
