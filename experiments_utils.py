import os

from omegaconf import DictConfig, ListConfig

from artifact_utils import write_yaml_artifact


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
        pass

    cpus_per_task = hydra_config.launcher.get("cpus_per_task")
    if cpus_per_task is not None:
        pass

    partition = hydra_config.launcher.get("partition")
    if partition is not None:
        pass
    write_yaml_artifact(
        "system_info.yaml",
        {
            "hostname": os.uname()[1],
            "mem_per_cpu": mem_per_cpu,
            "cpus_per_task": cpus_per_task,
            "partition": partition,
        },
    )


def log_params_from_omegaconf_dict(params, only_keys=None):
    data = {}
    for param_name, element in params.items():
        if only_keys is None or param_name in only_keys:
            data[param_name] = _collect_recursive(element)
    write_yaml_artifact("params.yaml", data)


def _collect_recursive(element):
    if isinstance(element, DictConfig):
        result = {}
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                result[k] = _collect_recursive(v)
            else:
                result[k] = v
        return result
    elif isinstance(element, ListConfig):
        return [_collect_recursive(v) if isinstance(v, (DictConfig, ListConfig)) else v for v in element]
    else:
        return element
