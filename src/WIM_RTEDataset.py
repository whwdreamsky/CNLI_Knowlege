from RTEDataset import RTEDataset,shuffle_arrays





class WIM_RTEDataset(RTEDataset):
    def __init__(self, sentences1, sentences2, sizes1, sizes2, labels, sentences1_lemma,
                 sentences2_lemma,sent1_charseq,sent2_charseq):

        RTEDataset.__init__(self,sentences1, sentences2,sizes1, sizes2, labels, sentences1_lemma,sentences2_lemma)

        #super(RTEDataset,self).\
        #    __init__(self, sentences1, sentences2,sizes1, sizes2, labels,None, sentences1_lemma,sentences2_lemma)
        self.sentences1_lemma = sentences1_lemma
        self.sentences2_lemma = sentences2_lemma
        self.sent1_charseq = sent1_charseq
        self.sent2_charseq = sent2_charseq

    def shuffle_data(self):
        """
        Shuffle all data using the same random sequence.
        :return:
        """
        shuffle_arrays(self.sentences1, self.sentences2, self.sentences1_lemma, self.sentences2_lemma,
                           self.sizes1, self.sizes2, self.labels,self.sent1_charseq,self.sent2_charseq)

    def get_batch(self, from_, to):
        """
        Return an RTEDataset object with the subset of the data contained in
        the given interval. Note that the actual number of items may be less
        than (`to` - `from_`) if there are not enough of them.

        :param from_: which position to start from
        :param to: which position to end
        :return: an RTEDataset object
        """
        if from_ == 0 and to >= self.num_items:
            return self

        subset = WIMRTEDataset(self.sentences1[from_:to],
                            self.sentences2[from_:to],
                            self.sizes1[from_:to],
                            self.sizes2[from_:to],
                            self.labels[from_:to],
                            self.sentences1_lemma[from_:to],
                            self.sentences2_lemma[from_:to],
                            self.sent1_charseq[from_:to],
                            self.sent2_charseq[from_:to]
                               )

        return subset
