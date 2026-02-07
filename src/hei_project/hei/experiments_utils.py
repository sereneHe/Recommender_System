import os

import mlflow
from mlflow import log_param
from omegaconf import DictConfig, ListConfig


def _create_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        strings = []
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                strings.append(_create_recursive(f'{parent_name}.{k}', v))
            else:
                strings.append(f"{parent_name}.{k} = '{v}'")
        return ' and '.join(strings)
    elif isinstance(element, ListConfig):
        assert False, 'Lists not supported - cannot make OR query'
    else:
        return f"{parent_name} = '{element}'"

def create_mlflow_query_string(params: DictConfig, finished=True):
    query_strings = []
    if finished:
        query_strings.append("status = 'FINISHED'")
    for param_name, element in params.items():
        query_strings.append(_create_recursive(f'params.{param_name}', element))
    return ' and '.join(query_strings)


def log_system_info(hydra_config: DictConfig):
    mem_per_cpu = hydra_config.launcher.get("mem_per_cpu")
    if mem_per_cpu is not None:
        log_param('slurm.mem_per_cpu', mem_per_cpu)

    cpus_per_task = hydra_config.launcher.get("cpus_per_task")
    if cpus_per_task is not None:
        log_param('slurm.cpus_per_task', cpus_per_task)

    partition = hydra_config.launcher.get("partition")
    if partition is not None:
        log_param('slurm.partition', partition)
    log_param('system.hostname', os.uname()[1])


def log_params_from_omegaconf_dict(params, only_keys=None):
    for param_name, element in params.items():
        if only_keys is None or param_name in only_keys:
            _explore_recursive(param_name, element)

def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f'{parent_name}.{k}', v)
            else:
                mlflow.log_param(f'{parent_name}.{k}', v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f'{parent_name}.{i}', v)
    else:
        mlflow.log_param(parent_name, element)
