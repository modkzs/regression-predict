# -*- coding: utf-8 -*-
from sklearn.ensemble import AdaBoostRegressor
from sklearn import clone

__author__ = 'yixuanhe'

class MultiAdaBoostRegressor(AdaBoostRegressor):
    def __init__(self,
                 base_estimators=[],
                 n_estimators=50,
                 learning_rate=1.,
                 loss='linear',
                 random_state=None):

        super(MultiAdaBoostRegressor, self).__init__(
            base_estimator=base_estimators[0],
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            loss=loss)
        self.times = 0
        self.base_estimators = base_estimators

    def _make_estimator(self, append=True):
        base_estimator_ = self.base_estimators[self.times % len(self.base_estimators)]
        estimator = clone(base_estimator_)
        estimator.set_params(**dict((p, getattr(self, p))
                                    for p in self.estimator_params))

        if append:
            self.estimators_.append(estimator)
            self.times += 1

        return estimator
