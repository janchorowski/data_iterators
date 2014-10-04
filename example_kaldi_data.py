'''
These are classes that I uilt on top of the new dataset/iterator framework.

Created on Aug 5, 2014

@author: chorows
'''

# Note: look into scripts/generate-python-datasets.py to see how to prepare data for the datasets

# Q: how large would a dense design matrix be?
# A: assume 40*11 coeffs per window (like 40 lda feats and splice 11 frames)
#    a window comes every 10ms, or 100 windows per second, 6000 windows per minute
#    1h of training data thus has 158.4e6 coefficients, or 5GB.
#
# Thus: dense design matrix would be a bad choice... but we will store stuff as a huge hdf5 file with blosc compression and hope for the best...

import logging
logger = logging.getLogger(__file__)

import os

import functools
from itertools import izip

from collections import OrderedDict, defaultdict

#import crcmod

import numpy as np
import scipy.weave

import tables

import pylearn2.datasets.dense_design_matrix
from pylearn2.space import VectorSequenceSpace, IndexSequenceSpace,\
    VectorSpace, IndexSpace

import cPickle as pickle

import kaldi_io

from pyspeech.dataset import SimpleDatasetScaffolding, NaturallyIndexedFiniteDataset
from pyspeech.dataset.iterators import AbstractDataIterator, TransformingIterator, LimitBatchSizeIterator, \
    ShuffledExamplesIterator, DataIterator, DataSpaceConformingIterator,\
    AbstractWrappedIterator
    
from pyspeech.dataset.multiproc_dataset import MutiprocessingDatasetMixin


def _make_numpy_aliased_array(arr, nrows):
    """
    Return a view of arr with neighboring rows sliced by chaging the shape while keeping the strides constant.
    
    Note that in the new array the elements are not indepenendent. The view is NOT writable and NOT C nor Fortran contiguous.
    
    >>> arr = np.arange(1.,11.).reshape(-1,1).dot(np.array([[1,10,100]]))
    >>> arr
    [[    1.    10.   100.]
     [    2.    20.   200.]
     [    3.    30.   300.]
     [    4.    40.   400.]
     [    5.    50.   500.]
     [    6.    60.   600.]
     [    7.    70.   700.]
     [    8.    80.   800.]
     [    9.    90.   900.]
     [   10.   100.  1000.]]
    >>> arr2 = make_numpy_aliased_array(arr,2)
    >>> arr2
    [[    1.    10.   100.     2.    20.   200.     3.    30.   300.]
     [    2.    20.   200.     3.    30.   300.     4.    40.   400.]
     [    3.    30.   300.     4.    40.   400.     5.    50.   500.]
     [    4.    40.   400.     5.    50.   500.     6.    60.   600.]
     [    5.    50.   500.     6.    60.   600.     7.    70.   700.]
     [    6.    60.   600.     7.    70.   700.     8.    80.   800.]
     [    7.    70.   700.     8.    80.   800.     9.    90.   900.]
     [    8.    80.   800.     9.    90.   900.    10.   100.  1000.]]
    >>> arr[2,0]=-20
    >>> arr2
    [[    1.    10.   100.     2.    20.   200.   -20.    30.   300.]
     [    2.    20.   200.   -20.    30.   300.     4.    40.   400.]
     [  -20.    30.   300.     4.    40.   400.     5.    50.   500.]
     [    4.    40.   400.     5.    50.   500.     6.    60.   600.]
     [    5.    50.   500.     6.    60.   600.     7.    70.   700.]
     [    6.    60.   600.     7.    70.   700.     8.    80.   800.]
     [    7.    70.   700.     8.    80.   800.     9.    90.   900.]
     [    8.    80.   800.     9.    90.   900.    10.   100.  1000.]]
self.readers['utterance_names'] = dict((n,n) for n in self.feature_names)
        self.sources['utterance_names'] = None
    
    """
    arr = np.ascontiguousarray(arr.view())

    assert len(arr.shape)==2
    assert nrows < arr.shape[0]
    
    #abuse the shape/stride relationship for fun and profit
    code = """
    #line 100 "generate-python-datasets.py"
    npy_intp* dims = PyArray_DIMS(arr_array);
    dims[0]-=nrows;
    dims[1]+=nrows*dims[1];
    PyArray_UpdateFlags(arr_array,  
        NPY_C_CONTIGUOUS | NPY_ALIGNED | NPY_F_CONTIGUOUS);
    arr_array->flags &= ~NPY_WRITEABLE;
    //PyArray_CLEARFLAGS(arr_array, NPY_WRITEABLE);
    """
    scipy.weave.inline(code, ["arr", "nrows"])
    return arr


class UtteranceIterator(AbstractDataIterator):
    """
    Iterates over features and targets read off Kaldi's .ark files
    """
    default_order = ('utterance_names', 'features', 'targets')
    
    def __init__(self, feats_rx, targets_rx=None, sources = ('utterance_names', 'features', 'targets')):
        """
            Read the kaldi data streams given by feats_rx and targets_rx
        """
        super(UtteranceIterator,self).__init__(source_names=sources)
        self.reorder = tuple(UtteranceIterator.default_order.index(s) for s in sources)
        
        self.feats_rdr = kaldi_io.SequentialBaseFloatMatrixReader(feats_rx)  
        if 'targets' in sources:
            assert(targets_rx is not None)
            self.targets_rdr = kaldi_io.TransRA(
                    kaldi_io.RandomAccessInt32VectorReader(targets_rx),
                    lambda x: x.reshape(-1,1))
        else:
            self.targets_rdr = defaultdict(lambda : None)
            
    def next(self):
        try:
            utt_name, utt_feats = self.feats_rdr.next()
            utt_targets = self.targets_rdr[utt_name]
            ret_inorder = (utt_name, utt_feats, utt_targets)
            return self.make(ret_inorder[i] for i in self.reorder)
        except StopIteration:
            if self.feats_rdr.is_open():
                self.feats_rdr.close()
                if hasattr(self.targets_rdr, 'close'): self.targets_rdr.close()
            raise

class ShuffledUtteranceIterator(AbstractDataIterator):
    """
    Iterates over features and targets read off in a random order Kaldi's .ark files
    """
    default_order = ('utterance_names', 'features', 'targets')
    
    def __init__(self, feats_rx, targets_rx, sources = ('utterance_names', 'features', 'targets'), rng=np.random):
        """
            Read the shuffled kaldi data streams given by feats_rx and targets_rx
        """
        super(ShuffledUtteranceIterator,self).__init__(source_names=sources)
        self.reorder = tuple(UtteranceIterator.default_order.index(s) for s in sources)
        
        utt_names = []
        with kaldi_io.SequentialInt32VectorReader(targets_rx) as rdr:
            for utt_name, unused_contents in rdr:
                utt_names.append(utt_name)
        
        rng.shuffle(utt_names)
        self.utt_names = utt_names
        
        self.feats_rdr = kaldi_io.RandomAccessBaseFloatMatrixReader(feats_rx)        
        self.targets_rdr = kaldi_io.TransRA(
             kaldi_io.RandomAccessInt32VectorReader(targets_rx),
             lambda x: x.reshape(-1,1))
       
    def next(self):
        if self.utt_names:
            utt_name = self.utt_names.pop()
            utt_feats = self.feats_rdr[utt_name]
            utt_targets = self.targets_rdr[utt_name]
            utt_name = np.array([[utt_name]])
            ret_inorder = (utt_name, utt_feats, utt_targets)
            return self.make(ret_inorder[i] for i in self.reorder)
        else:
            if self.feats_rdr.is_open():
                self.feats_rdr.close()
                if hasattr(self.targets_rdr, 'close'): self.targets_rdr.close()
            raise  StopIteration

class GroundHogPaddedBatchIterator(AbstractWrappedIterator):
    """
        Fetches pool_size sequences form the iterator, sorts by length and creates batches of similarly long utterances along with their masks.
        Optionally can limit total memory consumption of batch (many sshort utterances or a few longs)
    """
    def __init__(self, iterator, batch_size, pool_size, fill_values, 
                 max_num_frames=None, **kwargs):
        assert pool_size>=batch_size
        super(GroundHogPaddedBatchIterator, self).__init__(iterator=iterator, **kwargs)
        new_names = []
        for source_name in self.source_names:
            new_names.append(source_name)
            new_names.append(source_name + '_mask')
        self.OutputClass = AbstractDataIterator.create_output_class(new_names)
        self.properties['batch_size'] = batch_size
        self._pool_size = pool_size
        self._max_num_frames = max_num_frames
        self._len_getter = np.vectorize(lambda x: x.shape)
        self._fill_vals = [fill_values[s] for s in iterator.source_names]
        self.pool = []
        
    def next(self):
        if self.pool:
            return self.pool.pop()
        #try to fill the pool
        batch_queue = []
        batch_shapes = []
        
        batch_size = self.batch_size
        pool_size = self._pool_size
        len_getter = self._len_getter
        fill_vals = self._fill_vals
        max_num_frames = self._max_num_frames
        
        for seq, unused_i in izip(self.iterator, xrange(pool_size)):
            batch_queue.append(seq)
            batch_shapes.append(seq.features.shape[0]) #get the utterance length
        if not batch_queue:
            raise StopIteration
        sorted_idx = np.argsort(batch_shapes)
        if max_num_frames is None: #split by number of examples
            batch_idxs = np.split(sorted_idx, range(batch_size,len(batch_queue), batch_size))
        else:
            batch_idxs = []
            cur_idx = []
            for i in sorted_idx:
                batch_i_frames = batch_shapes[i]
                #here we use the fact that we have sorted by size-the larger the i, the larger the total batch
                if batch_i_frames*(len(cur_idx)+1) >= max_num_frames or len(cur_idx) >= batch_size:
                    #start a new batch
                    batch_idxs.append(cur_idx)
                    cur_idx = []
                cur_idx.append(i)
            if cur_idx:
                batch_idxs.append(cur_idx)
                    
        for batch_idx in batch_idxs:
            batch_seqs = np.take(batch_queue, batch_idx,0)
            seq_lens, seq_dims = len_getter(batch_seqs)
            max_len = seq_lens.max(0)
            max_dim = seq_dims.max(0)
            assert np.all(seq_dims==max_dim)
            batch_data = []
            batch_size = len(batch_idx)
            for src_id in xrange(batch_seqs.shape[1]):
                data = np.zeros((max_len[src_id], batch_size, max_dim[src_id]), dtype=batch_seqs[0,src_id].dtype) + fill_vals[src_id]
                mask = np.zeros((max_len[src_id], batch_size), dtype=np.float32) 
                for ex_id in xrange(batch_seqs.shape[0]):
                    data[:seq_lens[ex_id,src_id], ex_id, :] = batch_seqs[ex_id,src_id]
                    mask[:seq_lens[ex_id,src_id], ex_id] = 1
                if data.dtype.name.startswith('int'):
                    data.shape=data.shape[:2] #skip the unitary third dim
                batch_data.append(data)
                batch_data.append(mask) 
            self.pool.append(self.make(batch_data))
        if self.pool:
            return self.pool.pop()
        else:
            raise StopIteration


class InfiniteGroundHogIterator(object):
    def __init__(self, get_iterator_fun, rng=np.random, reset_rng=False,
                 inifinite_loop=True, **kwargs):
        super(InfiniteGroundHogIterator, self).__init__(**kwargs)
        self.get_iterator_fun = get_iterator_fun
        if reset_rng:
            self.rng_state = rng.get_satte()
        else:
            self.rng_state = None
        self.rng = rng
        self.peeked_batch = None
        self.inifinite_loop = inifinite_loop
        self.reset()
        self.next_offset = -1
        
    def reset(self):
        if self.rng_state is not None:
            self.rng.set_state(self.rng_state)
        self.iterator = self.get_iterator_fun(self.rng)

    def start(self,next_offset):
        pass
    
    def __iter__(self):
        return self

    def next(self, peek=False):
        if self.peeked_batch is not None:
            batch = self.peeked_batch
            if not peek:
                logger.debug("Use peeked batch")
                self.peeked_batch = None
            else:
                logger.debug("Repeeked at peeked batch")
            return batch
        try:
            raw_batch = self.iterator.next()
        except StopIteration:
            if not self.inifinite_loop:
                raise
            self.reset()
            raw_batch = self.iterator.next()
        batch = dict(x=raw_batch.features,
                     x_mask = raw_batch.features_mask,
                     y=raw_batch.targets,
                     y_mask=raw_batch.targets_mask)
        if peek:
            logger.debug("Set peeked batch")
            self.peeked_batch = batch
        return batch

def compute_spliced_frames(feats, splice_frames, return_contiguous=True):
    nrows, ncols = feats.shape
    feats_spliced = np.zeros((nrows+splice_frames-1, ncols), dtype=feats.dtype)
    feats_spliced[(splice_frames-1)/2: nrows+(splice_frames-1)/2,:] = feats
    feats_spliced = _make_numpy_aliased_array(feats_spliced, splice_frames-1)
    if return_contiguous:
        feats_spliced = np.ascontiguousarray(feats_spliced)
    return feats_spliced

def SplicedUtteranceIterator(iterator, splice_frames=1, return_contiguous=True):
    if splice_frames==1 or 'features' not in iterator.source_names:
        return iterator
    else:
        #capture the arguments in a function
        def transform(feats):
            return compute_spliced_frames(feats, splice_frames, return_contiguous)
        return TransformingIterator(iterator, transforms=dict(features=transform))

class KaldiSequences(NaturallyIndexedFiniteDataset):
    def __init__(self, data_dir, name='train', **kwargs):
        super(KaldiSequences,self).__init__(**kwargs)
        self.subset_name = name
        self.data_dir = data_dir

        feats_fn = os.path.join(self.data_dir, name + '-feats')
        sizes_fn = os.path.join(self.data_dir, name + '-sizes')
        
        self.readers =  OrderedDict()
        
        with kaldi_io.SequentialPythonReader('ark:%(sizes_fn)s.ark' % locals()) as sizes_reader:
            self.utterance_shapes = OrderedDict(sizes_reader)
        
        self.feature_names = np.array(self.utterance_shapes.keys())
        
        self.readers['features'] = kaldi_io.RandomAccessBaseFloatMatrixReader('scp:%(feats_fn)s.scp' % locals())
        self.sources['features'] = VectorSequenceSpace(self.utterance_shapes[self.feature_names[0]][1], 
                                                       dtype=kaldi_io.KALDI_BASE_FLOAT().name)
        
        targets_fn = os.path.join(self.data_dir, name + '-targets')
        
        if os.path.isfile(targets_fn+'.scp'):
            with open(os.path.join(self.data_dir, 'num-targets'),'r') as f:
                self._num_targets = int(f.read())
            self.readers['targets'] = kaldi_io.TransRA(kaldi_io.RandomAccessInt32VectorReader('scp:%(targets_fn)s.scp' % locals()),
                                                       lambda x: x.reshape(-1,1))
            self.sources['targets'] = IndexSequenceSpace(max_labels=self._num_targets, dim=1, dtype='int32')
        
        self.readers['utterance_names'] = dict((n,n) for n in self.feature_names)
        self.sources['utterance_names'] = None
        
        #crcfun = crcmod.predefined.mkPredefinedCrcFun('crc64-jones')
        self.readers['example_id'] = dict((n,np.array([[i]], dtype='int32')) for i,n in enumerate(self.feature_names))
        self.sources['example_id'] = VectorSpace(dim=1, dtype='int32')
        
        # Defaults for iterators
        self._iter_batch_size = 1
        self._iter_sources = ('features', 'targets')
        
    def close(self):
        #logger.info("Closing readers for data dir %s", self.data_dir)
        for reader in self.readers.itervalues():
            if hasattr(reader, 'close'):
                reader.close()
    
    def __del__(self):
        self.close()
    
    @functools.wraps(NaturallyIndexedFiniteDataset._validate_batch_size)
    def _validate_batch_size(self, batch_size):
        if batch_size!=1:
            raise Exception("KaldiFeatureSequences supports only batches of 1 sequence!")
    
    @functools.wraps(NaturallyIndexedFiniteDataset.get_num_examples)
    def get_num_examples(self):
        return len(self.feature_names)
    
    @functools.wraps(NaturallyIndexedFiniteDataset._get_batch)
    def _get_batch(self, example_indices, source_names):
        utt_name, = self.feature_names[example_indices]
        return tuple(self.readers[source][utt_name] for source in source_names)

    def utterance_iterator(self, mode=None, rng=None, sources = ('features','targets')):        
        # return self.get_iterator(mode=mode, batch_size=1, sources=sources, rng=rng)
        # 
        # Keep it separate since if we change the space we will run into problems with whatever depends on it
        rng = self._resolve_rng(mode, rng)
        subset_iterator = self._resolve_index_iterator(mode, batch_size=1, rng=rng)
        
        def generator():
            for batch_inidces in subset_iterator:
                utt_name, = self.feature_names[batch_inidces]
                yield tuple(self.readers[source][utt_name] for source in sources)
        ret =  DataIterator(generator(), source_names=sources)
        
        ret.properties['num_examples'] =  sum(s[0] for s in self.utterance_shapes.itervalues())
        ret.properties['stochastic'] = subset_iterator.stochastic
        return ret

class KaldiSplicedUtterances(SimpleDatasetScaffolding):
    def __init__(self, data_dir, subset='train', splice_frames=1, 
                 shuffling_memory=1000e6,
                 **kwargs):
        super(KaldiSplicedUtterances, self).__init__(**kwargs)
        self.sequences = KaldiSequences(data_dir, subset)
        self.splice_frames = splice_frames
        self.shuffling_memory = shuffling_memory
        
        seq_feat_spc = self.sequences.sources['features']
        self.sources['features'] = VectorSpace(dim=self.splice_frames*seq_feat_spc.dim , 
                                               sparse=False, dtype=seq_feat_spc.dtype)
        
        if self.sequences.has_targets():
            trgt_feat_spc = self.sequences.sources['targets']
            self.sources['targets'] = IndexSpace(max_labels=trgt_feat_spc.max_labels, dim=trgt_feat_spc.dim, 
                                                 dtype=trgt_feat_spc.dtype)
         
    def get_num_examples(self):
        return sum(s[0] for s in self.sequences.utterance_shapes.itervalues())
    
    def _get_iterator(self, mode, batch_size, desired_sources, rng):
        rename_map = OrderedDict((ds[0], ds[1][0]) for ds in desired_sources.iteritems())
        dest_sources = OrderedDict((ds[0], ds[1][1]) for ds in desired_sources.iteritems())
        iter_sources = set(rename_map.itervalues())
        iter_sources = dict((it_s, self.sources[it_s]) for it_s in iter_sources)
        
        utt_iterator = self.sequences.utterance_iterator(
                                mode=mode, rng=rng, sources=iter_sources.keys())
        
        utt_iterator = TransformingIterator(utt_iterator, transforms=dict(targets=lambda Y: Y.reshape(-1,1)))
        utt_iterator = SplicedUtteranceIterator(utt_iterator, 
                                                self.splice_frames, 
                                                return_contiguous=False)
         
        if mode in ['sequential']:
            utt_iterator = LimitBatchSizeIterator(utt_iterator, batch_size=batch_size)
        elif mode in ['random_uniform', 'shuffled_sequential']:
            if rng is None:
                rng = self.rng
            utt_iterator = ShuffledExamplesIterator(utt_iterator, batch_size=batch_size, 
                                    shuffling_mem=self.shuffling_memory, rng=rng) 
        else:
            raise Exception('KaldiSplicedUtterances only know how to return ' 
                            'a sequential or shuffled_sequential, or random_uniform iterator. Sorry :(')
        
        utt_iterator = DataSpaceConformingIterator(utt_iterator, dest_sources, iter_sources)
        return utt_iterator

class KaldiSplicedUtterancesMP(MutiprocessingDatasetMixin, KaldiSplicedUtterances):
    def __init__(self, *args, **kwargs):
        super(KaldiSplicedUtterancesMP,self).__init__(*args, **kwargs)

class KaldiFeatures(pylearn2.datasets.dense_design_matrix.DenseDesignMatrixPyTables):
    _default_seed = (17, 2, 946)
     
    def __init__(self, h5file, subset='train', rng=_default_seed):
        if type(h5file) is type(''):
            h5file = tables.openFile(h5file, mode='r')
        elif isinstance(h5file, KaldiFeatures):
            h5file=h5file.h5file
         
        self.h5file = h5file
        self.data = h5file.get_node('/data/'+subset)
        self.subset_name = subset
         
        X = self.data.features
        y = None
        self.num_targets = None
        if 'targets' in self.data:
            y = self.data.targets
            self.num_targets = h5file.root.data.num_targets[0]
        pylearn2.datasets.dense_design_matrix.DenseDesignMatrixPyTables.__init__(self, X=X, y=y, y_labels=self.num_targets, axes=('b',0), rng=rng)
     
    def _utterance_iterator(self, rng, sources):
        utt_index = self.data.utt_index
        for utt in utt_index.itersorted(utt_index.cols.name):
            ret = []
            for src in sources:
                if src=='utterance_names':
                    ret.append(utt['name'])
                elif src=='features':
                    ret.append(self.X[utt['beg']:utt['end'],...])
                elif src=='targets':
                    ret.append(self.y[utt['beg']:utt['end'],...])
                else:
                    raise KeyError('source: %s not found' % src)
            yield tuple(ret)
     
    def utterance_iterator(self, rng=None, sources = ('utterance_names','features','targets')):
        return DataIterator(self._utterance_iterator(rng, sources), sources)
     
    def get_subset(self, subset, rng=_default_seed):
        return KaldiFeatures(self.h5file, subset=subset, rng=rng)
     
    def get_test_set(self):
        return self.get_subset('test')
     
    def close(self):
        pass #do nothing as the table may be shared...
        #self.h5file.close()
