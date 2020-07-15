import sys
sys.path.append('..')

from pytorch_lr_scheduler_visualizer.lr_scheduler.SchedulerBuilder import SchedulerBuilder


class State:
    def __init__(self):
        self.sched_builder = SchedulerBuilder()
        self.schedulers = self.get_init_schedulers()
        self.logscale = False

    def get_init_schedulers(self):
        return [
            {'ExponentialLR': [
                {
                    'param': 'gamma',
                    'type': 'float',
                    'min': 0.0,
                    'max': 1.0,
                    'default': 0.9
                },
            ]},
            {'ReduceLROnPlateau': [
                {
                    'param': 'factor',
                    'type': 'float',
                    'min': 0.0,
                    'max': 1.0,
                    'default': 0.2
                },
                {
                    'param': 'patience',
                    'type': 'int',
                    'min': 0,
                    'max': 100,
                    'default': 10
                },
                {
                    'param': 'min_lr',
                    'type': 'float',
                    'min': 0.0,
                    'max': 0.1,
                    'default': 0.0
                },
            ]},
            {'CosineAnnealingLR': [
                {
                    'param': 'T_max',
                    'type': 'int',
                    'min': 1,
                    'max': 150,
                    'default': 10
                },
                {
                    'param': 'eta_min ',
                    'type': 'float',
                    'min': -0.05,
                    'max': 0.05,
                    'default': 0.0
                },
            ]},
        ]
