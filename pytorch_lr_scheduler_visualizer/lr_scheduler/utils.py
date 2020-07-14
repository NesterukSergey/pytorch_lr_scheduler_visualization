import matplotlib.pyplot as plt
from pytorch_lr_scheduler_visualizer.lr_scheduler.SchedulerBuilder import SchedulerBuilder


def get_all_default(iterations=100):
    sched_build = SchedulerBuilder()
    default_results = {}

    for scheduler_name in sched_build.supported_schedulers:
        params = sched_build.get_schedulers_params()[scheduler_name]
        default_params = {}
        for param in params:
            default_params[param['param']] = param['default']

        scheduler = sched_build.get_scheduler(scheduler_name, default_params)
        scheduler.iterate(100)

        default_results[scheduler_name] = scheduler.lr_history

    return default_results


def draw_all_default(logscale=True):
    default_results = get_all_default()

    for name in default_results:
        if logscale:
            plt.yscale('log')

        plt.title(name)
        plt.plot(range(len(default_results[name])), default_results[name])
        plt.show()
