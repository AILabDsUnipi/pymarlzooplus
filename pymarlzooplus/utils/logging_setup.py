from collections import defaultdict
import logging, os, re

import numpy as np
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback
from rich.table import Table
from rich.panel import Panel
from sacred.utils import apply_backspaces_and_linefeeds


class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_sacred = False
        self.use_wandb = False
        self.use_hdf = False

        self.tb_logger = None
        self._run_obj = None
        self.sacred_info = None
        self.wandb = None

        self.stats = defaultdict(lambda: [])

    def setup_tb(self, directory_name):
        # Import here so it doesn't have to be installed if you don't use it
        from tensorboard_logger import configure, log_value
        configure(directory_name)
        self.tb_logger = log_value
        self.use_tb = True

    def setup_sacred(self, sacred_run_dict):
        self._run_obj = sacred_run_dict
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def setup_wandb(self, args, config: dict, map_name: str, results_dir: str):
        """
        Initialize a wandb run if ``args.wandb.enable`` is truthy.

        All config-to-kwargs translation lives here so callers only need to
        forward the raw objects.

        Args:
            args: SimpleNamespace produced from the experiment config.
            config: the raw experiment config dict (``_config``).
            map_name: environment / map identifier used in the run name.
            results_dir: path to the Sacred results directory (used as the
                         default base for wandb file storage).
        """
        wandb_cfg = getattr(args, "wandb", None) if args is not None else None
        if not isinstance(wandb_cfg, dict) or not wandb_cfg.get("enable", False):
            return

        try:
            import wandb
        except Exception as e:
            self.console_logger.warning(
                f"wandb not available ({e}). Continuing without wandb."
            )
            self.use_wandb = False
            self.wandb = None
            return

        # ---- build wandb.init kwargs from the config ----
        wandb_kwargs: dict = {}

        # Run name
        if wandb_cfg.get("run_name"):
            wandb_kwargs["name"] = wandb_cfg["run_name"]
        else:
            alg_name = getattr(args, "learner", config.get("learner", ""))
            exp_name = config.get("name", "exp")
            seed = config.get("seed")
            seed_str = f"_seed{seed}" if seed is not None else ""
            wandb_kwargs["name"] = f"{exp_name}_{map_name}_{alg_name}{seed_str}"

        # Scalar optional fields
        for cfg_key, wb_key in [
            ("project", "project"),
            ("entity", "entity"),
            ("id", "id"),
            ("notes", "notes"),
            ("tags", "tags"),
            ("sync_tensorboard", "sync_tensorboard"),
            ("resume", "resume"),
        ]:
            value = wandb_cfg.get(cfg_key)
            if value is not None and value != "" and value != []:
                wandb_kwargs[wb_key] = value

        # Save directory
        try:
            wb_base = wandb_cfg.get("save_dir", "results/wandb")
            wandb_dir = wb_base if os.path.isabs(wb_base) else os.path.join(results_dir, wb_base)
            os.makedirs(wandb_dir, exist_ok=True)
            wandb_kwargs["dir"] = wandb_dir
        except Exception:
            pass

        # ---- init ----
        try:
            run = wandb.init(**wandb_kwargs, config=config)
            self.wandb = wandb
            self.use_wandb = True
            self._wandb_run = run
        except Exception as e:
            self.console_logger.warning(
                f"Failed to initialize wandb ({e}). Continuing without wandb."
            )
            self.use_wandb = False
            self.wandb = None

    def finish_wandb(self):
        """Gracefully close the wandb run, flushing any buffered data."""
        if getattr(self, "use_wandb", False) and getattr(self, "_wandb_run", None) is not None:
            try:
                self._wandb_run.finish()
            except Exception:
                pass
            self.use_wandb = False
            self.wandb = None
            self._wandb_run = None

    def log_stat(self, key, value, t, to_sacred=True):
        self.stats[key].append((t, value))

        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]

            self._run_obj.log_scalar(key, value, t)

        # Log to wandb if configured
        if getattr(self, "use_wandb", False) and getattr(self, "wandb", None) is not None:
            try:
                # wandb.log expects a dict and optionally a step
                self.wandb.log({key: float(self._to_float(value))}, step=int(t))
            except Exception:
                # Don't propagate wandb logging errors
                pass

    @staticmethod
    def _to_float(x):
        """Robustly convert numpy/torch scalars to float (NaN on failure)."""
        try:
            if hasattr(x, "item"):
                return float(x.item())
            return float(np.asarray(x).reshape(-1)[0])
        except Exception:
            return float("nan")

    def _get_rich_console(self):
        """Return the Rich Console from RichHandler if present, else None."""

        candidates = [self.console_logger, getattr(self.console_logger, "parent", None), logging.getLogger()]
        for lg in candidates:
            if not lg:
                continue
            for h in getattr(lg, "handlers", []):
                if h.__class__.__name__ == "RichHandler":
                    return getattr(h, "console", None)

        return None

    def print_recent_stats(self, default_window=5, epsilon_window=1):
        """
        Pretty logging of recent rolling stats.
        - default_window: rolling window for all metrics except 'epsilon'
        - epsilon_window: window for 'epsilon'
        - cols: columns in plain-text fallback
        - include_std: add a Std column (Rich only)
        """
        # Header values
        t_env, ep = self.stats["episode"][-1]

        # Collect (name, mean, std, window)
        items = []
        for k in sorted(self.stats.keys()):
            if k == "episode":
                continue
            window = epsilon_window if k.lower() == "epsilon" else default_window
            recent = self.stats[k][-window:]
            vals = [self._to_float(v) for _, v in recent] if recent else []
            mean_val = float(np.nanmean(vals)) if len(vals) else float("nan")
            std_val = float(np.nanstd(vals)) if len(vals) else float("nan")
            items.append((k, mean_val, std_val, window))

        # Render a nice table to the Rich console
        console = self._get_rich_console()
        assert console is not None, "Rich console not found in logger handlers."
        table = Table(show_header=True, header_style="bold")
        table.add_column("Metric", justify="left", no_wrap=True)
        table.add_column("Mean", justify="right")
        table.add_column("Std", justify="right")
        table.add_column("Average Window", justify="right")
        for name, mean_val, std_val, win in items:
            table.add_row(name, f"{mean_val:.4f}", f"{std_val:.4f}", f"{win}")
        header = f"Recent Stats  |  t_env: {t_env:,}  |  Episode: {ep}"
        console.print(Panel(table, title=header))


def get_logger(name="pymarlzooplus", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    install_rich_traceback(show_locals=False)
    rh = RichHandler(rich_tracebacks=True, markup=True, show_path=False, enable_link_path=False)
    rh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(rh)
    logger.propagate = False

    return logger

# Use this for Sacred's captured_out_filter (cleans prints + Rich ANSI if they’re captured)
def captured_filter(text: str) -> str:
    text = apply_backspaces_and_linefeeds(text)
    return re.compile(r"\x1b\[[0-9;]*[mK]").sub("", text)





