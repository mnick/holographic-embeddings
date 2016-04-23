#!/usr/bin/env python

import numpy as np
from base import Experiment, FilteredRankingEval
from skge.util import ccorr
from skge import StochasticTrainer, PairwiseStochasticTrainer, HolE
from skge import activation_functions as afs


class HolEEval(FilteredRankingEval):

    def prepare(self, mdl, p):
        self.ER = ccorr(mdl.R[p], mdl.E)

    def scores_o(self, mdl, s, p):
        return np.dot(self.ER, mdl.E[s])

    def scores_s(self, mdl, o, p):
        return np.dot(mdl.E, self.ER[o])


class ExpHolE(Experiment):

    def __init__(self):
        super(ExpHolE, self).__init__()
        self.parser.add_argument('--ncomp', type=int, help='Number of latent components')
        self.parser.add_argument('--rparam', type=float, help='Regularization for W', default=0)
        self.parser.add_argument('--afs', type=str, default='sigmoid', help='Activation function')
        self.evaluator = HolEEval

    def setup_trainer(self, sz, sampler):
        model = HolE(
            sz,
            self.args.ncomp,
            rparam=self.args.rparam,
            af=afs[self.args.afs],
            init=self.args.init
        )
        if self.args.no_pairwise:
            trainer = StochasticTrainer(
                model,
                nbatches=self.args.nb,
                max_epochs=self.args.me,
                post_epoch=[self.callback],
                learning_rate=self.args.lr,
                samplef=sampler.sample
            )
        else:
            trainer = PairwiseStochasticTrainer(
                model,
                nbatches=self.args.nb,
                max_epochs=self.args.me,
                post_epoch=[self.callback],
                learning_rate=self.args.lr,
                margin=self.args.margin,
                samplef=sampler.sample
            )
        return trainer

if __name__ == '__main__':
    ExpHolE().run()
