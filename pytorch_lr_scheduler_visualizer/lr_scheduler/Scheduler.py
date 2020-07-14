class Scheduler:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.lr_history = []
        self.name = self.scheduler.__class__.__name__

    def get_lr(self):
        return self.scheduler.optimizer.state_dict()['param_groups'][0]['lr']

    def iterate(self, iterations):
        self.lr_history.append(self.get_lr())
        if self.name in ['ReduceLROnPlateau']:
            for i in range(iterations):
                self.scheduler.step(0)  # simulate constant loss
                self.lr_history.append(self.get_lr())
        else:
            for i in range(iterations):
                self.scheduler.step()
                self.lr_history.append(self.get_lr())
