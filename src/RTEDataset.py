# coding =utf-8
import numpy as np
import abc
class RTEDataset(object):
    """
    Class for better organizing a data set. It provides a separation between
       first and second sentences and also their sizes.
    """
    abc.__metaclass__ = abc.ABCMeta


    def __init__(self, sentences1, sentences2,sizes1, sizes2, labels):
        """
        :param sentences1: A 2D numpy array with sentences (the first in each
            pair) composed of token indices
        :param sentences2: Same as above for the second sentence in each pair
        :param sizes1: A 1D numpy array with the size of each sentence in the
            first group. Sentences should be filled with the PADDING token after
            that point
        :param sizes2: Same as above
        :param labels: 1D numpy array with labels as integers
        """
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.sizes1 = sizes1
        self.sizes2 = sizes2
        self.labels = labels
        self.num_items = len(sentences1)

    def shuffle_data(self):
        """
        Shuffle all data using the same random sequence.
        :return:
        """
        shuffle_arrays(self.sentences1, self.sentences2,
                       self.sizes1, self.sizes2, self.labels)

    def get_batch(self, from_, to):
        """
        Return an RTEDataset object with the subset of the data contained in
        the given interval. Note that the actual number of items may be less
        than (`to` - `from_`) if there are not enough of them.

        :param from_: which position to start from
        :param to: which position to end
        :return: an RTEDataset object
        """
        subset = RTEDataset(self.sentences1[from_:to],
                                self.sentences2[from_:to],
                                self.sizes1[from_:to],
                                self.sizes2[from_:to],
                                self.labels[from_:to])
        return subset
        
def shuffle_arrays(*arrays):
    """
    Shuffle all given arrays with the same RNG state.

    All shuffling is in-place, i.e., this function returns None.
    """
    rng_state = np.random.get_state()
    for array in arrays:
        np.random.shuffle(array)
        np.random.set_state(rng_state)