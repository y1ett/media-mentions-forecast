PY ?= .venv/bin/python

.PHONY: venv all ingest build dq train tune predict monitor retrain rollback explain dashboard status test

venv:       ## окружение и зависимости
	python3 -m venv .venv || python3 -m virtualenv .venv
	.venv/bin/pip install -r requirements.txt

all:        ## полный цикл: загрузка -> витрина -> качество -> обучение/мониторинг -> прогноз
	$(PY) -m src

ingest:     ## инкрементальная загрузка всех источников (первый запуск ~1 ч из-за лимита HN)
	$(PY) -m src.ingest all

build:      ## staging и витрина mart.topic_daily
	$(PY) -m src.processing build

dq:         ## контроль качества данных
	$(PY) -m src.processing dq

status:     ## таблицы и журнал загрузок
	$(PY) -m src.processing status

train:      ## отбор признаков, сравнение моделей, реестр версий
	$(PY) -m src.train fit

tune:       ## автоподбор гиперпараметров LightGBM (Optuna)
	$(PY) -m src.tune

predict:    ## прогноз на 7 дней
	$(PY) -m src.train predict

rollback:   ## откат к предыдущей версии модели
	$(PY) -m src.train rollback

monitor:    ## дрейф, качество на новых фактах, правило переобучения
	$(PY) -m src.monitor

retrain:    ## мониторинг + переобучение, если сработало правило
	$(PY) -m src.monitor --auto-retrain

explain:    ## make explain TOPIC=ai_llm DATE=2026-10-01 H=1
	$(PY) -m src.explain --topic $(TOPIC) --date $(DATE) --horizon $(or $(H),1)

dashboard:  ## http://localhost:8501
	$(PY) -m streamlit run src/dashboard.py

test:
	$(PY) -m pytest -q tests
