# Retention Rate Analysis

Анализ удержания пользователей для оценки Product Market Fit дейтингового приложения.

![Retention Rate](retention_chart.png)

## Задача

Расчёт еженедельного Retention Rate - доли пользователей, продолжающих использовать приложение спустя N недель после регистрации.

## Данные

| Таблица | Описание |
|---------|----------|
| `retention_users` | user_id, username, registration_date |
| `retention_users_activity` | user_id, date, активность (logins, messages, likes и др.) |

## Метрика

```
Retention Rate = active_users / total_users
```

где `week` - порядковый номер недели с момента регистрации пользователя.

## Пост-хок и дополнительное

Добавить график-дашборд
Желательно бы такое на питоне тоже реализовать
