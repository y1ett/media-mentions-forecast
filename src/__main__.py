"""Полный цикл одной командой: python -m src (или make all).

загрузка -> staging/витрина -> контроль качества (критичные ошибки останавливают цикл) -> обучение (если модели
нет) или мониторинг с правилом переобучения -> прогноз -> мониторинг.
"""
import argparse
import logging
import subprocess
import sys
from pathlib import Path

from .config import load_config

log = logging.getLogger("pipeline")


def step(name: str, *args: str) -> int:
    log.info("=== %s ===", name)
    return subprocess.call([sys.executable, "-m", *args])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ingest", action="store_true")
    ap.add_argument("--force-train", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    if not a.skip_ingest:
        step("Загрузка данных", "src.ingest", "all")
    if step("Staging и витрина", "src.processing", "build"):
        return 1
    if step("Контроль качества данных", "src.processing", "dq"):
        log.error("критичные проблемы качества данных — цикл остановлен (см. meta.dq_results)")
        return 1
    if a.force_train or not (Path(cfg["paths"]["models"]) / "registry.json").exists():
        if step("Обучение", "src.train", "fit"):
            return 1
    else:
        step("Мониторинг и правило переобучения", "src.monitor", "--auto-retrain")
    if step("Прогноз", "src.train", "predict"):
        return 1
    step("Мониторинг", "src.monitor")
    log.info("готово: make dashboard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
