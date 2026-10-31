# Работа с репозиторием

[← О проекте](README.md)

## Проверка и публикация изменений

Из корня локального клона:

```bash
git switch -c docs/update-project
python -m unittest discover -s tests
git status --short

# Указать только файлы, относящиеся к изменению.
git add README.md docs/
git diff --cached --check
git diff --cached --stat
git commit -m "Update project documentation"
git push -u origin docs/update-project
```

Создать pull request через интерфейс GitHub или `gh pr create`. При изменениях кода добавить соответствующие файлы и тесты в staging явно. Не использовать force push для обновления основной ветки.

Данные, локальные модели и SQLite studies исключены из Git. Перед публикацией ноутбуков проверить, что выводы не содержат секретов или персональных путей. Небольшие воспроизводимые таблицы результатов можно оставлять для просмотра без запуска.

Описание эксперимента, ограничения и команды запуска находятся в [README](README.md) и [инструкции воспроизведения](docs/reproduction.md).
