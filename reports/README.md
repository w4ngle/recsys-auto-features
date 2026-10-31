# Артефакты экспериментов

[← О проекте](../README.md) · [Интерпретация результатов](../docs/results.md)

## Актуальная серия

- [final_report.pdf](final_report.pdf) — описание задачи, реализации, результатов и ограничений.
- [recheck/results_recheck.csv](recheck/results_recheck.csv) — метрики четырёх классификаторов.
- [recheck/bootstrap_recheck.json](recheck/bootstrap_recheck.json) — точечные разности, среднее bootstrap и 95% интервалы.
- [recheck/params_used.json](recheck/params_used.json) — параметры, применённые в контрольном запуске.
- [recheck/data_audit.csv](recheck/data_audit.csv) — размеры частей и число пользователей с позитивом.
- [recheck/run_metadata.json](recheck/run_metadata.json) — окружение и настройки запуска.
- [recheck/input_manifest.json](recheck/input_manifest.json) — SHA-256 входных parquet контрольной серии.

`importance_*.csv` содержат gain-важности признаков. Файлы обученных моделей сохраняются локально при выполнении ноутбука и не включены в Git.

Все метрики `recheck/` получены на сохранённых таблицах прежнего временного протокола. Они не относятся к пересборке с часовыми зазорами.

## Архив

`historical/`, прежние CSV/JSON в корне `reports/`, `full_explanation.pdf` и `progress_report.pdf` сохранены как история исследования. Они могут содержать промежуточные или устаревшие формулировки. Актуальные выводы приведены в README, `docs/` и `final_report.pdf`.

Ранние артефакты не редактируются задним числом. Новый запуск ноутбука перезаписывает `recheck/`; перед сравнением серий нужно сохранить отдельную копию и зафиксировать происхождение данных.
