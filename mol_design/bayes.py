from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import numpy as np
from bayes_opt.observer import _Tracker
from bayes_opt.event import Events


class Recorder(_Tracker):
    def __init__(self):
        super(Recorder, self).__init__()
        self.history = []
    def update(self, event, instance):
        if event == Events.OPTIMIZATION_STEP:
            data = dict(instance.res[-1])
            self.history.append( data )
        self._update_tracker(event, instance)

class BayesOpt:
    def __init__(self, x, target_function, mode="max"):
        self.x = x
        self.var_bounds = { }
        for i in range(len(x)):
            for j in range(len(x)):
                self.var_bounds[ f"a_{i}_{j}" ] = (0,1) if j > i else (0,0)
        self.hist = []
        self.target_function = target_function
        self.mode = mode
        #self.logger = Recorder()

    def optimize(self, **kwargs):
        optimizer = BayesianOptimization(self.target_function, self.var_bounds)
       # optimizer.subscribe(Events.OPTIMIZATION_STEP, self.logger)
        optimizer.maximize(**kwargs)
        self.hist = optimizer.res