#!/usr/bin/env python

import numpy as np
from base import Experiment, FilteredRankingEval
from skge import TransE, PairwiseStochasticTrainer


class TransEEval(FilteredRankingEval):

    def prepare(self, mdl, p):
        self.ER = mdl.E + mdl.R[p]

    def scores_o(self, mdl, s, p):
        return -np.sum(np.abs(self.ER[s] - mdl.E), axis=1)

    def scores_s(self, mdl, o, p):
        return -np.sum(np.abs(self.ER - mdl.E[o]), axis=1)


class ExpTransE(Experiment):

    def __init__(self):
        super(ExpTransE, self).__init__()
        self.parser.add_argument('--ncomp', type=int, help='Number of latent components')
        self.evaluator = TransEEval

    def setup_trainer(self, sz, sampler):
        model = TransE(sz, self.args.ncomp, init=self.args.init)
        trainer = PairwiseStochasticTrainer(
            model,
            nbatches=self.args.nb,
            margin=self.args.margin,
            max_epochs=self.args.me,
            learning_rate=self.args.lr,
            samplef=sampler.sample,
            post_epoch=[self.callback]
        )
        return trainer

if __name__ == '__main__':
    ExpTransE().run()
