from ..experiments.domain import DummyLRScheduler, OptimizerSchedulerBundle
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR
from torch.nn.modules import Module
from .custom_optimizers import RAdam


def noopt(model: Module,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        step_size: int = 10
        ) -> OptimizerSchedulerBundle:
    optimizer = SGD(model.parameters(), lr=0, momentum=0)
    scheduler = StepLR(optimizer, step_size=step_size)
    return OptimizerSchedulerBundle(optimizer=optimizer, scheduler=scheduler)


def lrs(model: Module,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        ) -> OptimizerSchedulerBundle:
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0, weight_decay=0)
    scheduler = StepLR(optimizer, step_size=10)
    return OptimizerSchedulerBundle(optimizer=optimizer, scheduler=scheduler)


def lrs_imgnet(model: Module,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        ) -> OptimizerSchedulerBundle:
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=10)
    return OptimizerSchedulerBundle(optimizer=optimizer, scheduler=scheduler)



def lrs60_imgnet(model: Module,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        ) -> OptimizerSchedulerBundle:
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=20)
    return OptimizerSchedulerBundle(optimizer=optimizer, scheduler=scheduler)



def lrs60(model: Module,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        step_size: int = 20
        ) -> OptimizerSchedulerBundle:
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size)
    return OptimizerSchedulerBundle(optimizer=optimizer, scheduler=scheduler)


def lrs90(model: Module,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        step_size: int = 30
        ) -> OptimizerSchedulerBundle:
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size)
    return OptimizerSchedulerBundle(optimizer=optimizer, scheduler=scheduler)



def lrs180(model: Module,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        step_size: int = 50
        ) -> OptimizerSchedulerBundle:
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size)
    return OptimizerSchedulerBundle(optimizer=optimizer, scheduler=scheduler)



def sgd(model: Module,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        ) -> OptimizerSchedulerBundle:
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    return OptimizerSchedulerBundle(optimizer=optimizer)


def adam(model: Module) -> OptimizerSchedulerBundle:
    optimizer = Adam(model.parameters())
    return OptimizerSchedulerBundle(optimizer=optimizer)


def adamw(model: Module) -> OptimizerSchedulerBundle:
    optimizer = AdamW(model.parameters())
    return OptimizerSchedulerBundle(optimizer=optimizer)


def radam(model: Module) -> OptimizerSchedulerBundle:
    optimizer = RAdam(model.parameters())
    return OptimizerSchedulerBundle(optimizer=optimizer)
