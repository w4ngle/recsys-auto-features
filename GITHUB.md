# Команды для GitHub

Команды ниже предназначены для существующего локального репозитория. Они создают новую ветку,
копируют исправленные файлы без удаления остальных и отправляют ветку в уже настроенный origin.
Модели LightGBM и parquet не добавляются в Git. Команды не запускались автоматически.

```bash
cd /Users/artur/Desktop/recsys-auto-features

# Сохранить текущие незакоммиченные правки отслеживаемых файлов.
git diff > ../recsys-before-final.patch
git status --short

git switch -c finalize-optuna-audit

FINAL='/Users/artur/Documents/Codex/2026-10-31/referenced-chatgpt-conversation-this-is-an/outputs/recsys-auto-features'
rsync -av --exclude='.git/' --exclude='data/' "$FINAL/" ./

source .venv/bin/activate
python -m unittest discover -s tests

git add .gitignore README.md GITHUB.md AUDIT.md requirements.txt requirements-checked.txt prepare_data.py src notebooks/T-Lab.ipynb notebooks/T-Lab.original.ipynb reports/historical reports/recheck reports/final_report.pdf

git diff --cached --stat
git diff --cached --check
git commit -m "Finalize Optuna results, notebook and experiment audit"
git push -u origin finalize-optuna-audit
```

Если установлен GitHub CLI, после push можно создать pull request:

```bash
gh pr create --title 'Finalize Optuna results and experiment audit' --body 'Adds a sequential verification notebook, separates historical and rerun results, corrects paired user bootstrap, documents Optuna and temporal protocol limitations, and supplies a cache rebuild with label-horizon gaps. Validation: unit checks and replay of four classifiers on existing caches. Historical studies and raw-data rebuild are not reproduced.'
```

Новый корректный временной протокол запускается отдельно; его метрики нельзя выдавать за прежние:

```bash
python prepare_data.py
export RECSYS_DATA_DIR="$PWD/data/rebuilt"
# Открыть notebooks/T-Lab.ipynb, выбрать Restart Kernel and Run All.
# Для нового Optuna-поиска: RUN_SEARCH=True и новый STUDY_TAG.
```
