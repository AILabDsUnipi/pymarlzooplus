import multiprocessing
import subprocess
from itertools import product
import yaml
import random

import click

_CPU_COUNT = multiprocessing.cpu_count() - 1


def _flatten_lists(_object):
    for item in _object:
        if isinstance(item, (list, tuple, set)):
            yield from _flatten_lists(item)
        else:
            yield item


def _compute_combinations(config_file, shuffle, seeds):
    config = yaml.load(config_file, Loader=yaml.FullLoader)
    combinations = []
    for k, v in config["grid-search"].items():
        if type(v) is not list:
            v = [v]
        combinations.append([f"{k}={v_i}" for v_i in v])

    group_comb = []
    for _, v in config["grid-search-groups"].items():
        d = {}
        for d_i in v:
            d.update(d_i)

        group_comb.append(tuple([f"{k}={v_i}" for k, v_i in d.items()]))
    combinations.append(group_comb)

    click.echo("Found following combinations: ")
    click.echo(click.style(" X ", fg="red", bold=True).join([str(s) for s in combinations]))

    configs = list(product(*combinations))
    configs = [list(_flatten_lists(c)) for c in configs]

    configs = [[f"hypergroup=hp_grp_{i}"] + c for i, c in enumerate(configs)]

    configs = list(product(configs, [f"seed={i}" for i in range(seeds)]))
    configs = [list(_flatten_lists(c)) for c in configs]

    if shuffle:
        random.Random(1337).shuffle(configs)

    return configs


def work(cmd):
    cmd = cmd.split(" ")
    return subprocess.call(cmd, shell=False)


@click.group()
def cli():
    pass


@cli.group()
@click.option("--config", type=click.File(), default="config.yaml")
@click.option("--shuffle/--no-shuffle", default=True)
@click.option("--seeds", default=3, show_default=True, help="How many seeds to run")
@click.pass_context
def run(ctx, config, shuffle, seeds):
    combos = _compute_combinations(config, shuffle, seeds)
    if len(combos) == 0:
        click.echo("No valid combinations. Aborted!")
        exit(1)
    ctx.obj = combos


@run.command()
@click.option(
    "--cpus",
    default=_CPU_COUNT,
    show_default=True,
    help="How many processes to run in parallel",
)
@click.pass_obj
def locally(combos, cpus):
    configs = ["python main.py " + " ".join([c for c in combo if c.startswith("--")]) + " with " + " ".join([c for c in combo if not c.startswith("--")]) for combo in combos]

    click.confirm(
        f"There are {click.style(str(len(combos)), fg='red')} combinations of configurations "
        "(taking into account the number of seeds). "
        f"Up to {cpus} will run in parallel. "
        "Continue?",
        abort=True,
    )
    pool = multiprocessing.Pool(processes=cpus)
    print(pool.map(work, configs))


@run.command()
@click.argument(
    "index", type=int,
)
@click.pass_obj
def single(combos, index):
    """Runs a single hyperparameter combination
    INDEX is the index of the combination to run in the generated combination list
    """

    config = combos[index]
    cmd = "python main.py " + " ".join([c for c in config if c.startswith("--")]) + " with " + " ".join([c for c in config if not c.startswith("--")])
    print(cmd)
    work(cmd)


if __name__ == "__main__":
    cli()
