"""HTTP-запросы с повторными попытками: таймауты, 429 (rate limit) с Retry-After, 5xx, обрывы соединения.
"""
import logging
import time

import requests

log = logging.getLogger("http")

try:
    # доверять системному хранилищу сертификатов (Windows/macOS): нужно, когда антивирус или корпоративный
    # прокси проверяет HTTPS и подставляет свой корневой сертификат (например, Kaspersky)
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass


class SourceUnavailable(Exception):
    """Источник не ответил после всех попыток или вернул недопустимый ответ."""


def get(url: str, *, params: dict | None = None, headers: dict | None = None,
        timeout: int = 30, retries: int = 4, backoff: float = 2.0) -> requests.Response:
    for attempt in range(1, retries + 1):
        wait = backoff ** attempt
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        except requests.exceptions.SSLError as exc:
            # повтор не поможет: сертификат подменён (антивирус/прокси) или не доверен
            raise SourceUnavailable(
                f"{url}: ошибка проверки SSL-сертификата ({exc.__class__.__name__}). Если включён антивирус с "
                "проверкой HTTPS (Kaspersky и т.п.), установите truststore (pip install truststore) или отключите "
                "проверку защищённых соединений для python.exe") from exc
        except (requests.ConnectionError, requests.Timeout) as exc:
            log.warning("%s: попытка %d/%d, сетевая ошибка: %s", url, attempt, retries, exc)
        else:
            if resp.status_code == 429:
                try:
                    wait = float(resp.headers.get("Retry-After", wait))
                except ValueError:
                    pass
                log.warning("%s: 429 Too Many Requests, ждём %.0f с", url, wait)
            elif resp.status_code >= 500:
                log.warning("%s: попытка %d/%d, HTTP %d", url, attempt, retries, resp.status_code)
            elif resp.status_code >= 400:
                raise SourceUnavailable(f"{url}: HTTP {resp.status_code}")
            else:
                return resp
        if attempt < retries:
            time.sleep(wait)
    raise SourceUnavailable(f"{url}: нет ответа после {retries} попыток")


def http_opts(cfg: dict) -> dict:
    """Общие параметры запросов из конфига (таймаут, повторы, User-Agent)."""
    h = cfg["http"]
    return {
        "timeout": h["timeout"],
        "retries": h["retries"],
        "backoff": h["backoff"],
        "headers": {"User-Agent": cfg["project"]["user_agent"]},
    }
