import numpy as np

from collections import OrderedDict, Counter

import functools
import warnings

from pylearn2.utils.rng import make_np_rng
from pylearn2.space import CompositeSpace
from pylearn2.utils import safe_izip
from pylearn2.utils.iteration import resolve_iterator_class

from iterators import AbstractDataIterator,\
    AbstractWrappedIterator, DataSpaceConformingIterator
from pylearn2.datasets.dataset import Dataset

class AbstractSimpleDataset(object):
    """
    Base class defining the interface to datasets.
    
    Important fields:
        - sources: an ordered dict nameing the data in this dataset and specifying the spaces of the data
    
    Similar to PyLearn2 datasets, defaults for the iterator function are saved as fields of this class.
    """
    def __init__(self, **kwargs):
        super(AbstractSimpleDataset, self).__init__(**kwargs)
        self.sources = OrderedDict() 
        self.sources.__doc__ = """
        A dict from source names to their default space 
        (which should correspond to how they are internally stored).
        """
    
        #defaults for iterators:
        self._iter_subset_class = 'sequential'
        self._iter_batch_size = 10
        
        #self._iter_sources = self.sources.keys() 
    
    def get_batch_design(self, batch_size=None, sources=None):
        """
        Get a batch of the desired size from the dataset. 
        
        Parameters
        ----------
        batch_size: how many exampels to return
        sources: either a list of strings that names the sources, a dict (or OrderedDict) 
            that maps sources to the desired output spaces. 
            note: to retrieve the same source multiple times, append a suffix separetd by '_' 
            to the source name.
            
        Examples
        --------
        get_batch_design(self, sources=dict(features_asarrays=VectorSpace..., features_asimages=Conv2dspace...)
        """
        raise NotImplementedError
    
    def get_iterator(self, mode=None, batch_size=None, sources=None, rng=None):
        """
        The parameters follow parameters of Dataset.iterator. I used a different function name, to be able to provide 
        an iterator function for iterator compatibility.
        
        Parameters
        ----------
        see doc for Dataset.iterator
        
        Returns
        -------
        An iterator derived from AbstractDataIterator
        
        Note
        ----
        Why no num_batches? The only reason I (JCh) see for it is to ease the splitting of the data into parts. 
        But this can be done ny an iterator selecting ceratin batches from a stream of them, which woult play more nicely with 
        streamed data.
        
        For practical reasons, one usually wants to constrain the memory used and to specify the batch size!
        """
        raise NotImplementedError
    
    def get_num_examples(self):
        """
        Return the number of examples, use float('inf') for an infinite streaming data set or float('nan') for a don't care value.
        """
        return float('nan')

    #utility functions
    def has_targets(self):
        return 'targets' in self.sources
    
    def has_features(self):
        return 'features' in self.sources

class _UntuplingIterator(AbstractWrappedIterator):
    """ Return single element instead of one-element tuple
    
    Note: this is not a true AbstractDataIterator, as we violate the protocol. It is for compatibility with pylearn2 only.
    """
    def next(self):
        ret, = self.iterator.next()
        return ret

class SimpleDatasetScaffolding(AbstractSimpleDataset, Dataset):
    """
    A class that eases the implementation of AbstractSimpleDatasets. 
    It also provides a compatibility layer with Dataset.Dataset 
    
    Since Dataset interface assumes that a single source can be returned multiple times with multiple spaces and 
    AbstractDataIterator assumes that each element of the bach has a unique name, 
    name mangling is employed to separate between instances of the same source but in different spaces.  
    
    The main purpose of this class is to have all the code that resolves iteration modes, sizes and similar details. 
    Subclasses can expect that all parameters of _get_iterator are provided and have simple implementations.
    
    Derived classes must provide an implementation of _get_iterator
    
    Control of the behavior of iteration can be realized by overriding the functions:
        - _validate_mode to prohibit using some iteration modes
        - _validate_batch_size to validate the requested batch size
        - _resolve* to control filling of specific defaults for get_iterator
    """
    _default_seed = (17, 2, 946)
    
    def __init__(self, rng=_default_seed, **kwargs):
        super(SimpleDatasetScaffolding, self).__init__(**kwargs)
        self._rng = make_np_rng(rng, which_method="random_integers")
        
        #self.sources['example_id']
        #TODO: add example id support??   
    
    #These abstract functions are required for proper operation
    def _get_iterator(self, mode, batch_size, desired_sources, rng):
        """
        Get the iterator. Mode is still passed as a string, for datasets for which it is difficult to index by numbers
        
        Sources is an OrderedDict from desired source names to pairs of source names, desired spaces 
        """
        raise NotImplementedError
    
    #These functions can be overriden for finer-grain control of   what happens
    def _validate_mode(self, mode):
        """
        Should throw an exception if the mode is not supported by this dataset
        """
        pass
    
    def _validate_batch_size(self, batch_size):
        """
        Should throw an exception if the batch_size is not supported by this dataset
        """
        pass
    
    def _is_mode_stochastic(self, mode):
        iterator = resolve_iterator_class(mode)
        return iterator.stochastic
    
    def _resolve_rng(self, mode, rng):
        if self._is_mode_stochastic(mode):
            if rng is None:
                rng = self._rng
        return rng
    
    def _resolve_sources(self, sources):
        if not sources:
            sources = getattr(self, '_iter_sources', self.sources)
        if not sources:
            sources = self.sources
        if type(sources)==str:
            warnings.warn("Sources should always be a tuple, even if only one source is requested")
            sources = (sources,)
        return sources
    
    def _resolve_mode(self, mode):
        if mode is None:
            mode = getattr(self,'_iter_subset_class','sequential')
        return mode

    def _resolve_batch_size(self, batch_size):
        if batch_size is None:
            batch_size = getattr(self,'_iter_batch_size',10)
        return batch_size

    #This is the Dataset.Dataset compatibility layer
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None, data_specs=None,
                 return_tuple=False):
        if num_batches is not None:
            batch_size = np.ceil(float(self.get_num_examples() / float(num_batches)))
            if not np.isfinite(batch_size):
                raise ValueError("Can compute the batch size from numbatches in an infinite dataset")
            
        if topo is not None or targets is not None:
            if data_specs is not None:
                raise ValueError('In DenseDesignMatrix.iterator, both the '
                                 '"data_specs" argument and deprecated '
                                 'arguments "topo" or "targets" were '
                                 'provided.',
                                 (data_specs, topo, targets))

            warnings.warn("Usage of `topo` and `target` arguments are "
                          "being deprecated, and will be removed "
                          "around November 7th, 2013. `data_specs` "
                          "should be used instead.",
                          stacklevel=2)

            # build data_specs from topo and targets if needed
            if topo is None:
                topo = getattr(self, '_iter_topo', False)
            if topo:
                raise ValueError("Don't know what you mean by a topo batch. Please swithc to use the new data_specs interface! ")
            else:
                feature_space = self.sources['features']

            if targets is None:
                targets = getattr(self, '_iter_targets', False)
            if targets:
                target_space = self.sources['tagrets']
                space = CompositeSpace((feature_space, target_space))
                source = ('features', 'targets')
            else:
                space = feature_space
                source = 'features'

            data_specs = (space, source)
        
        #TODO: call resolve_sources
        if data_specs is None:
            data_specs = (CompositeSpace(self.sources.values()), self.sources.keys())
        
        desired_space, desired_source = data_specs
        if isinstance(desired_space, CompositeSpace):
            desired_space = desired_space.components
            desired_source = desired_source
        else:
            desired_space = (desired_space,)
            desired_source = (desired_source,)
        
        source_repetitions = Counter(desired_source)
        name_mangling=Counter()
        
        converted_sources = OrderedDict()
        for source, space in safe_izip(desired_source, desired_space):
            if source_repetitions[source]>1:
                name_mangling[source] += 1
                source = '%s_%d' % (source, name_mangling[source])
            converted_sources[source] = space
        
        iterator = self.get_iterator(mode, batch_size, converted_sources, rng)
        if return_tuple:
            return iterator
        else:
            return _UntuplingIterator(iterator)
        
    #This is normal function implementation
    @functools.wraps(AbstractSimpleDataset.get_iterator)
    def get_iterator(self, mode=None, batch_size=None, sources=None, rng=None):
        mode = self._resolve_mode(mode)
        self._validate_mode(mode)
        batch_size = self._resolve_batch_size(batch_size)
        self._validate_batch_size(batch_size)
        
        rng = self._resolve_rng(mode, rng)
        sources = self._resolve_sources(sources)
        desired_sources = OrderedDict()
        
        if isinstance(sources, dict):
            for source, space in sources.iteritems():
                if source in self.sources:
                    desired_sources[source] = (source,space)
                else:
                    possible_source_name = "".join(source.split('_')[:-1])
                    if possible_source_name in self.sources:
                        desired_sources[source] = (possible_source_name, space)
        else:
            for source in sources:
                space = self.sources[source]
                desired_sources[source] = (source,space)

        return self._get_iterator(mode, batch_size, desired_sources, rng)

    @functools.wraps(AbstractSimpleDataset.get_batch_design)
    def get_batch_design(self, batch_size=None, sources=None):
        iterator = self.get_iterator('sequential', batch_size, sources)
        return iterator.next()

class _IndexedIterator(AbstractDataIterator):
    """
    Helper class to return batches formed of examples enumerated by an index iterator
    """
    def __init__(self, dataset, source_names, idx_iterator, **kwargs):
        super(_IndexedIterator, self).__init__(source_names=source_names, **kwargs)
        self.properties['sources'] = dataset.sources
        self._dataset = dataset
        self._idx_iterator=idx_iterator.__iter__()
    
    def next(self):
        idx = self._idx_iterator.next()
        ret = self.make(self._dataset._get_batch(idx, self.source_names))
        return ret
    
class NaturallyIndexedFiniteDataset(SimpleDatasetScaffolding):
    """
    A helper to implement datasets for which it is easy to fetch examples by their ID number.
    
    It can serve as a basis for e.g. a reworked DenseDesignMatrix. Subclasses only need to provide _get_batch method
    which request ezeampes by their id.
    """
    def __init__(self, **kwargs):
        super(NaturallyIndexedFiniteDataset, self).__init__(**kwargs)
    
    # Mandatory overrides    
    def _get_batch(self, example_indices, source_names):
        raise NotImplementedError
    
    # Normal operation
    def _resolve_index_iterator(self, mode, batch_size, rng):
        mode = resolve_iterator_class(mode)
        return mode(self.get_num_examples(),
                 batch_size, 
                 None, #num_batches 
                 rng=rng)

    def _get_iterator(self, mode, batch_size, desired_sources, rng):
        idx_iterator = self._resolve_index_iterator(mode, batch_size, rng)
        rename_map = OrderedDict()
        dataset_source_names = set()
        destination_data_specs = OrderedDict()
        for dest_src_name, (data_src_name, dest_space) in desired_sources.iteritems():
            rename_map[dest_src_name]=data_src_name
            dataset_source_names.add(data_src_name)
            destination_data_specs[dest_src_name] = dest_space
        dataset_iter = _IndexedIterator(dataset=self, source_names=dataset_source_names, idx_iterator=idx_iterator)
        return DataSpaceConformingIterator(dataset_iter, destination_data_specs, rename_map=rename_map)
