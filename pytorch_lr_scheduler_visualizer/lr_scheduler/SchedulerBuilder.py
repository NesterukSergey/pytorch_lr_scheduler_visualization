import torch
from pytorch_lr_scheduler_visualizer.lr_scheduler.Scheduler import Scheduler


class SchedulerBuilder:
    def __init__(self):
        self.max_iter = 10000
        self.default_lr = 1e-3

        self.schedulers_params = self.get_schedulers_params()
        self.supported_schedulers = self.get_supported_schedulers()

    def get_schedulers_params(self):
        return {
            'CosineAnnealingLR': [
                {
                    'param': 'T_max',
                    'type': 'int',
                    'min': 1,
                    'default': self.max_iter
                },
                {
                    'param': 'eta_min ',
                    'type': 'float',
                    'min': 0.0,
                    'max': 1.0,
                    'default': 0.0
                },
                {
                    'param': 'last_epoch',
                    'type': 'int',
                    'min': -1,
                    'max': self.max_iter,
                    'default': -1
                },
            ],
            #             'CyclicLR': [
            #                 {
            #                     'param': 'base_lr',
            #                     'type': 'float',
            #                     'min':0.0,
            #                     'max': 1.0,
            #                     'default': self.default_lr
            #                 },
            #                 {
            #                     'param': 'max_lr',
            #                     'type': 'float',
            #                     'min':0.0,
            #                     'max': 1.0,
            #                     'default': 1.0
            #                 },
            #                 {
            #                     'param': 'step_size_up',
            #                     'type': 'int',
            #                     'min': 1,
            #                     'max': self.max_iter,
            #                     'default': 2000
            #                 },
            #                 {
            #                     'param': 'step_size_down',
            #                     'type': 'int',
            #                     'min': 1,
            #                     'max': self.max_iter,
            #                     'default': 2000
            #                 },
            #                 {
            #                     'param': 'mode',
            #                     'type': 'str',
            #                     'optioms': ['triangular', 'triangular2', 'exp_range'],
            #                     'default': 'triangular'
            #                 },
            #                 {
            #                     'param': 'gamma',
            #                     'type': 'float',
            #                     'min':0.0,
            #                     'max': 1.0,
            #                     'default': 1.0
            #                 },
            #                 {
            #                     'param': 'base_momentum',
            #                     'type': 'float',
            #                     'min':0.0,
            #                     'max': 1.0,
            #                     'default': 0.8
            #                 },
            #                 {
            #                     'param': 'max_momentum',
            #                     'type': 'float',
            #                     'min':0.0,
            #                     'max': 1.0,
            #                     'default': 0.9
            #                 },
            #                 {
            #                     'param': 'last_epoch',
            #                     'type': 'int',
            #                     'min': -1,
            #                     'max': self.max_iter,
            #                     'default': -1
            #                 },

            #             ],
            'ReduceLROnPlateau': [
                {
                    'param': 'factor',
                    'type': 'float',
                    'min': 0.0,
                    'max': 1.0,
                    'default': 0.1
                },
                {
                    'param': 'patience',
                    'type': 'int',
                    'min': 0,
                    'max': self.max_iter,
                    'default': 10
                },
                {
                    'param': 'min_lr',
                    'type': 'float',
                    'min': 0.0,
                    'max': 1.0,
                    'default': 0.0
                },
                {
                    'param': 'eps',
                    'type': 'float',
                    'min': 0.0,
                    'max': 1.0,
                    'default': 1e-8
                },
            ],
            'ExponentialLR': [
                {
                    'param': 'gamma',
                    'type': 'float',
                    'min': 0.0,
                    'max': 1.0,
                    'default': 0.9
                },
                {
                    'param': 'last_epoch',
                    'type': 'int',
                    'min': -1,
                    'max': self.max_iter,
                    'default': -1
                },
            ],
            'MultiStepLR': [
                {
                    'param': 'milestones',
                    'type': 'list',
                    'default': [30, 80]
                },
                {
                    'param': 'gamma',
                    'type': 'float',
                    'min': 0.0,
                    'max': 1.0,
                    'default': 0.1
                },
                {
                    'param': 'last_epoch',
                    'type': 'int',
                    'min': -1,
                    'max': self.max_iter,
                    'default': -1
                },
            ],
            #             'OneCycleLR': [
            #                 {
            #                     'param': 'max_lr',
            #                     'type': 'float',
            #                     'min':0.0,
            #                     'max': 1.0,
            #                     'default': 1.0
            #                 },
            #                 {
            #                     'param': 'total_steps',
            #                     'type': 'int',
            #                     'min': 0,
            #                     'max': self.max_iter,
            #                     'default': 1000
            #                 },
            #                 {
            #                     'param': 'pct_start',
            #                     'type': 'float',
            #                     'min':0.1,
            #                     'max': 0.9,
            #                     'default': 0.3
            #                 },
            # #                 {
            # #                     'param': 'anneal_strategy',
            # #                     'type': 'str',
            # #                     'optioms': ['cos', 'linear'],
            # #                     'default': 'cos'
            # #                 },
            #                 {
            #                     'param': 'base_momentum',
            #                     'type': 'float',
            #                     'min':0.0,
            #                     'max': 1.0,
            #                     'default': 0.85
            #                 },
            #                 {
            #                     'param': 'max_momentum',
            #                     'type': 'float',
            #                     'min':0.0,
            #                     'max': 1.0,
            #                     'default': 0.95
            #                 },
            #                 {
            #                     'param': 'div_factor',
            #                     'type': 'float',
            #                     'min':0.0,
            #                     'max': 1.0,
            #                     'default': 1e-4
            #                 },
            #                 {
            #                     'param': 'last_epoch',
            #                     'type': 'int',
            #                     'min': -1,
            #                     'max': self.max_iter,
            #                     'default': -1
            #                 },

            #             ],
            'StepLR': [
                {
                    'param': 'step_size',
                    'type': 'int',
                    'min': 1,
                    'max': self.max_iter,
                    'default': 5
                },
                {
                    'param': 'gamma',
                    'type': 'float',
                    'min': 0.0,
                    'max': 1.0,
                    'default': 0.1
                },
                {
                    'param': 'last_epoch',
                    'type': 'int',
                    'min': -1,
                    'max': self.max_iter,
                    'default': -1
                },
            ],
        }

    def get_supported_schedulers(self):
        return list(self.schedulers_params.keys())

    def check_scheduler_name(self, name):
        if name not in self.supported_schedulers:
            raise Exception('This scheduler is not supported. Pick one of:', self.supported_schedulers)

    def get_scheduler_params(self, name):
        self.check_scheduler_name(name)
        return self.schedulers_params[name]

    def parse_params(self, obj):
        s = ''
        for i in obj.keys():
            s += i + '=' + str(obj[i]) + ','

        return s[:-1]

    def get_scheduler(self, name, params):
        self.check_scheduler_name(name)
        model = torch.nn.Linear(1, 1)  # dummmy model
        optimizer = torch.optim.Adam(model.parameters())  # dummy optimizer

        params = self.parse_params(params)
        d = {'torch': torch,
             'optimizer': optimizer}
        code = 'scheduler = torch.optim.lr_scheduler.{}(optimizer, {})'.format(name, params)
        exec(code, d)
        scheduler = d['scheduler']

        return Scheduler(scheduler)
