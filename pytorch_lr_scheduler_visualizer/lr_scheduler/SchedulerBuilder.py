import torch
from pytorch_lr_scheduler_visualizer.lr_scheduler.Scheduler import Scheduler


class SchedulerBuilder:
    def __init__(self):
        self.max_iter = 500
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
            ],
            'ReduceLROnPlateau': [
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
                    'max': self.max_iter,
                    'default': 10
                },
                {
                    'param': 'min_lr',
                    'type': 'float',
                    'min': 0.0,
                    'max': 0.1,
                    'default': 0.0
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
            ],
            'StepLR': [
                {
                    'param': 'step_size',
                    'type': 'int',
                    'min': 1,
                    'max': 100,
                    'default': 5
                },
                {
                    'param': 'gamma',
                    'type': 'float',
                    'min': 0.0,
                    'max': 1.0,
                    'default': 0.1
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
        model = torch.nn.Linear(1, 1)  # dummy model
        optimizer = torch.optim.Adam(model.parameters())  # dummy optimizer

        params = self.parse_params(params)
        d = {'torch': torch,
             'optimizer': optimizer}
        code = 'scheduler = torch.optim.lr_scheduler.{}(optimizer, {})'.format(name, params)
        exec(code, d)
        scheduler = d['scheduler']

        return Scheduler(scheduler)
