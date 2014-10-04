data_iterators
==============

Sandbox for data iteration schemes

Quick summary:

We extend the Python iterator protocol (mandating only `__iter__` and `next` methods) by requiring also that:
  1. Each yield returns a `namedtuple` representing one batch of data
  2. The iterator hase a `source_names` property that lists the names of atch elements. 
     These names correspond to the fields of the returned namedtuples.
  3. The following optional properties are defined, with the following defaults:
     - sources - returns an OrderedDict mapping source_names to data Spaces (or raises Expetion if 
       Spaces information is not present)
     - num\_examples, num\_batches, batch\_size, uneven, stochastic properties, ising `nan` as a placeholder when a numerical           value is not know and `None` if another value is not known.

Since the iterators return named data, we always refer to the data by their names (e.g. batch.features instead of batch[0]) which makes the code more readable and less error-prone is multiple sources are combined into the same batch

The specialized iterators are built that: transform data, shuffle and divide examples in batches, convert between data spaces and even fetch the data in a subprocess.

The iterators are used to define datasets: The get_iterator method is supposed to return them.

For datasets a hierarchy of classes is implemented:
  1. AbstractSimpleDataset which defines the interface of the datasets
  2. SimpleDatasetScaffolding that factores out common argument parsing of get_iterator and provides PyLearn2 dataset compatibility
  3. NaturallyIndexedFiniteDataset which further assumes that examples can easily be enumerated and fetched by their id (netural number)

Examples of usage are in the file example\_kaldi\_data.py that I used during my experiments.
