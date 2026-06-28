import importlib.util
import os.path as osp


def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    module_name = f"{__name__}.{osp.splitext(name)[0]}"
    spec = importlib.util.spec_from_file_location(module_name, pathname)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
