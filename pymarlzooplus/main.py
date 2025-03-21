import sys
import os
from os.path import dirname, abspath
import collections
from copy import deepcopy
import inspect
import yaml

from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import numpy as np
import torch as th

from pymarlzooplus.utils._logging import get_logger
from pymarlzooplus.run import run

SETTINGS['CAPTURE_MODE'] = "fd"  # set to "no" if you want to see stdout/stderr in the console
logger = get_logger()


def generate_seed():
    # Generate 16 bytes (128 bits) of cryptographically strong randomness
    secure_bytes = os.urandom(16)
    # Convert those bytes to a (potentially large) integer
    seed_value = int.from_bytes(secure_bytes, byteorder="big")
    # Ensure it's within the 32-bit range (0 <= seed < 2**32)
    return seed_value % (2**32)


def my_main(_run, _config, _log):

    ## Setting the random seed throughout the modules
    config = config_copy(_config)
    # If a seed is provided, use it, otherwise use the autogenerated one
    if 'seed' in config['env_args']:
        if config['env_args']['seed'] is None:
            config['env_args']['seed'] = generate_seed()
        config['seed'] = config['env_args']['seed']
    else:
        if config['seed'] is None:
            config['seed'] = generate_seed()
        config['env_args']['seed'] = config['seed']
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])

    # run the framework
    run(_run, config, _log)


def get_caller_path():
    frame = inspect.stack()
    return dirname(abspath(frame[-1].filename))


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)

        if arg_name == "--env-config":
            for param in params:
                if 'env_args.seed' in param:
                    if '=' in params[params.index(param)]:
                        config_dict['env_args']['seed'] = int(params[params.index(param)].split('=')[1])
                    else:
                        assert ' ' in params[params.index(param)], \
                            f"'env_args.seed': {params[params.index(param)]}"
                        config_dict['env_args']['seed'] = int(params[params.index(param)].split(' ')[1])
                    break

        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def format_params(params_dict):
    """
    Convert a dictionary of parameters into a list of command-line arguments.

    Args:
        params_dict (dict): Dictionary with parameter keys and values.

    Returns:
        list
    """
    args = []

    for key, value in params_dict.items():
        if key != "env_args" and key != "algo_args" and "env-config" not in key:
            args.append(f"--{key}={value}")

    # Keep the "algo_args" as a dict if exists
    algo_params = None
    if "algo_args" in params_dict:
        assert isinstance(params_dict["algo_args"], dict), f"type(params_dict): {type(params_dict)}"
        algo_params = params_dict["algo_args"]

    # Add the "--env-config=..." if exists
    for key, value in params_dict.items():
        if "env-config" in key:
            args.append(f"--{key}={value}")

    # Add the "env_args" if exists
    if "env_args" in params_dict and isinstance(params_dict["env_args"], dict):
        args.append("with")
        for key, value in params_dict["env_args"].items():
            args.append(f"env_args.{key}={value}")

    return args, algo_params


def pymarlzooplus(params):
    try:
        algo_params = None
        if isinstance(params, dict):
            params, algo_params = format_params(params)
            params = ["pymarlzooplus/main.py"] + params[:]

        th.set_num_threads(1)

        # Get the defaults from default.yaml
        with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "default.yaml error: {}".format(exc)

        # Load algorithm and env base configs
        env_config = _get_config(params, "--env-config", "envs")
        alg_config = _get_config(params, "--config", "algs")
        config_dict = recursive_dict_update(config_dict, env_config)
        config_dict = recursive_dict_update(config_dict, alg_config)
        if algo_params is not None:
            # Parameters from the command line override the default ones and the ones from the config file
            config_dict = recursive_dict_update(config_dict, algo_params)

        try:
            map_name = config_dict["env_args"]["map_name"]
        except:
            map_name = config_dict["env_args"]["key"]

        # Create a fresh Sacred experiment
        ex = Experiment("pymarlzooplus")
        ex.logger = logger
        ex.captured_out_filter = apply_backspaces_and_linefeeds

        # now add all the config to sacred
        ex.add_config(config_dict)

        for param in params:
            if param.startswith("env_args.map_name"):
                map_name = param.split("=")[1]
            elif param.startswith("env_args.key"):
                map_name = param.split("=")[1]

        # Save to disk by default for sacred
        logger.info("Saving to FileStorageObserver in results/sacred.")
        results_path = os.path.join(get_caller_path(), "results")
        file_obs_path = os.path.join(results_path, "sacred", f"{config_dict['name']}", f"{map_name}")

        ex.observers = [(FileStorageObserver(file_obs_path))]
        ex.main(my_main)
        ex.run_commandline(params)
    except Exception as e:
        raise RuntimeError(f"pymarlzooplus failed with error:\n{e}") from e


if __name__ == '__main__':
    _params = deepcopy(sys.argv)
    pymarlzooplus(_params)

