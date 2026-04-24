import os.path
import yaml

REPO_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def _load_config() -> dict:
    cfg = {}
    for name in ("config.yaml", "config.local.yaml"):
        path = os.path.join(REPO_DIR, name)
        if os.path.exists(path):
            with open(path) as f:
                cfg.update(yaml.safe_load(f) or {})
    return cfg

_cfg = _load_config()
DATASET_DIR: str = _cfg["dataset_dir"]
ALL_GCL_TABLE: str = _cfg["all_gcl_table"]