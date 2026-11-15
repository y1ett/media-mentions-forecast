"""Загрузка config.yaml с переопределением через переменные окружения / .env.
"""
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

# переменная окружения -> путь к ключу в config.yaml
ENV_OVERRIDES = {
    "MMF_DB": ("paths", "db"),
    "MMF_DATA_DIR": ("paths", "data_dir"),
    "CONTACT_EMAIL": ("project", "contact"),
    "HTTP_TIMEOUT": ("http", "timeout"),
    "HTTP_RETRIES": ("http", "retries"),
    "HISTORY_START": ("sources", "start_date"),
    "HN_RPS": ("sources", "hn", "requests_per_sec"),
    "TUNING_TRIALS": ("model", "tuning", "n_trials"),
    "MMF_MODELS": ("paths_models",),
}


def _read_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def _parse(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def load_config(path: Path | None = None) -> dict:
    _read_dotenv(ROOT / ".env")
    cfg = yaml.safe_load((path or ROOT / "config.yaml").read_text(encoding="utf-8"))
    for env, keys in ENV_OVERRIDES.items():
        if os.environ.get(env):
            value = os.environ[env]
            node = cfg
            for key in keys[:-1]:
                node = node[key]
            node[keys[-1]] = _parse(value)
    for key in ("data_dir", "db"):
        p = Path(cfg["paths"][key])
        cfg["paths"][key] = str(p if p.is_absolute() else ROOT / p)
    p = Path(cfg.get("paths_models", "models"))
    cfg["paths"]["models"] = str(p if p.is_absolute() else ROOT / p)
    cfg["project"]["user_agent"] = cfg["project"]["user_agent"].format(
        contact=cfg["project"]["contact"]
    )
    return cfg
