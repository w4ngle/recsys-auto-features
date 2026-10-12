# Автоматизация генерации признаков в рекомендательных системах

Учебный проект: можно ли заменить ручной подбор признаков (например, окно «7 дней») автоматическим поиском, 
который не проиграет хорошо затюненному градиентному бустингу.

**Данные:** [Yambda-50M](https://huggingface.co/datasets/yandex/yambda) (Яндекс.Музыка: прослушивания и лайки).
**Модель:** LightGBM. **Метрики:** AUC, NDCG@10, Recall@10 (внутри каждого пользователя).

> Статус: ручной бейзлайн собран, автоматизация признаков в работе. Подробный разбор для новичка: [`reports/progress_report.pdf`](reports/progress_report.pdf).

## Идея

1. Собрать сильный бустинг на ручных признаках — это «планка».
2. Описать признак шаблоном `(сущность × группировка × окно × агрегат × событие)`, 
   сгенерировать сотни комбинаций и оставить те, что улучшают метрику на валидации.
3. Показать цифрами, что найденные признаки (например, окно 10 дней вместо 7) не хуже ручных.

## Постановка

- Строка = прослушивание `(uid, item_id, timestamp)`. `label = 1`, если пользователь лайкнул трек в течение часа после начала прослушивания (доля позитивов около 0,8%).
- Подвыборка: 2000 случайных пользователей со всей историей (10,15 млн прослушиваний, 173 тыс. лайков).
- Сплит по времени: train, затем val (14 дней), затем test (14 дней).
- **Без утечек:** признаки считаются только по событиям строго до среза, в коде есть `assert`.
- Из кандидатов убраны пары, которые пользователь уже лайкал до среза (новым лайком они стать не могут).

## Признаки

| Группа | Примеры |
|---|---|
| user (`u_`) | прослушивания/лайки за 1, 7, 30 дней и всё время, конверсия, средняя доля дослушивания |
| item (`i_`) | то же по треку |
| user × item (`ui_`) | сколько раз юзер слушал/лайкал именно этот трек |
| user × artist / album (`uar_`, `ual_`) | сколько раз юзер слушал/лайкал треки этого артиста/альбома |

Окна задаются списком `WINDOWS` в `src/features.py`; именно его подбирает автоматизация.

## Результаты (на текущий момент)

| Модель | Выборка | AUC | NDCG@10 | Recall@10 |
|---|---|---|---|---|
| random | test | 0,496 | 0,044 | 0,055 |
| top-popular | test | 0,494 | 0,032 | 0,042 |
| boosting full, 4 среза | test | 0,860 | 0,087 | 0,119 |

Бейзлайны посчитаны до фильтра «уже лайкнуто» и будут пересчитаны. Прямое сравнение пока приблизительное.

TODO: boosting minimal / full manual / full + artist & album на одних данных, Optuna, автоматический поиск признаков, абляции, несколько сидов.

## Структура

```
src/
  data.py       загрузка Yambda, подвыборка юзеров, метки, сплит по времени
  features.py   ручные признаки (build, add_indirect, build_train)
  metrics.py    AUC, NDCG@k, Recall@k
  models.py     обучение LightGBM
notebooks/      эксперименты (T-Lab.ipynb)
reports/        PDF-отчёт
data/           данные (в git не попадают)
```

## Как запустить

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# macOS: для LightGBM нужен OpenMP
brew install libomp
```

Минимальный пример:

```python
from src.data import *
from src.features import *
from src.models import *
from src.metrics import report

listens, likes = load_yambda("listens.parquet"), load_yambda("likes.parquet")
L, K = sample_users(listens, likes, n_users=2000)
L = make_labels(L, K)
maps = {"artist_id": load_mapping("artist_item_mapping.parquet"),
        "album_id":  load_mapping("album_item_mapping.parquet")}
L, K = add_groups(L, K, maps)

t_val, t_test, t_end = time_split(L)
Xtr = build_train(L, K, t_val, maps=maps)
Xva = add_indirect(L, K, build(L, K, window(L, t_val, t_test), t_val), t_val, maps)
Xte = add_indirect(L, K, build(L, K, window(L, t_test, t_end + 1), t_test), t_test, maps)
Xva, Xte = [X[X.ui_likes_all == 0] for X in (Xva, Xte)]

fit_eval("full + artist/album", feature_columns(Xtr), Xtr, Xva, Xte)
```
