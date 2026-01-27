with user_weeks as (
    -- определяем номер недели активности относительно даты регистрации каждого пользователя
    select a.user_id,
           toInt64(floor(dateDiff('day', u.registration_date, a.date) / 7)) as week_number
    from default.retention_users_activity a
             join default.retention_users u on a.user_id = u.user_id),

     weekly_active as (
         -- считаем уникальных активных пользователей по каждой неделе
         select week_number             as week,
                count(distinct user_id) as active_users
         from user_weeks
         where week_number > 0 -- исключаем нулевую неделю (неделя регистрации)
         group by week_number),

     total as (
         -- общее количество зарегистрированных пользователей
         select count(*) as total_users
         from default.retention_users)

select w.week,
       w.active_users,
       t.total_users,
       round(w.active_users / t.total_users, 3) as retention_percentage
from weekly_active w
         cross join total t
order by w.week
