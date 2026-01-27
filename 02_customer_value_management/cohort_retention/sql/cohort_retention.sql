-- определяем когорту каждого пользователя по месяцу регистрации
with cohorts as (select user_id,
                        extract(month from reg_date) ::int as cohort_id
                 from cr_users
                 where reg_date >= '2024-01-01'
                   and reg_date < '2025-01-01'),

-- считаем размер каждой когорты (сколько всего зарегистрировалось в каждом месяце)
     cohort_sizes as (select cohort_id,
                             count(*) as total_users
                      from cohorts
                      group by cohort_id),

-- находим последний месяц с активностью в данных
     max_month as (select max(extract(month from action_date)) ::int as max_m
                   from cr_user_actions
                   where action_date >= '2024-01-01'
                     and action_date < '2025-01-01'),

-- генерируем все пары когорта-месяц (от месяца регистрации до последнего месяца)
     cohort_months as (select cs.cohort_id,
                              m.month_id
                       from cohort_sizes cs
                                cross join lateral generate_series(cs.cohort_id, (select max_m from max_month)) as m(month_id)
    ),

-- определяем активных пользователей: открыл приложение + использовал ключевую функцию
    monthly_activity as (
select
    c.user_id, c.cohort_id, extract (month from a.action_date):: int as month_id
from cohorts c
    join cr_user_actions a
on c.user_id = a.user_id
where a.action_date >= '2024-01-01' and a.action_date < '2025-01-01'
group by c.user_id, c.cohort_id, extract (month from a.action_date)
-- условие активности: открытие приложения + любое ключевое действие
having
    sum (case when a.action_type = 'open_app' then 1 else 0 end)
     > 0
   and sum (case when a.action_type in ('manage_notes'
     , 'manage_plants'
     , 'set_reminder'
     , 'view_articles'
     , 'share_plants') then 1 else 0 end)
     > 0
    )
     ,

-- считаем количество активных пользователей в каждой когорте по месяцам
    active_counts as (
select
    cohort_id, month_id, count (distinct user_id) as active_users
from monthly_activity
group by cohort_id, month_id
    )

select cm.cohort_id,
       trim(to_char(to_date(cm.cohort_id::text, 'MM'), 'Month')) || ' users' as cohort_name,
       cm.month_id,
       coalesce(ac.active_users, 0)                                          as active_users,
       round(coalesce(ac.active_users, 0) * 100.0 / cs.total_users, 1)       as retention_rate
from cohort_months cm
         join cohort_sizes cs on cm.cohort_id = cs.cohort_id
         left join active_counts ac on cm.cohort_id = ac.cohort_id and cm.month_id = ac.month_id
order by cm.cohort_id, cm.month_id;
