with users as (select distinct user_id
               from default.churn_submits),
     days as (select distinct toDate(timestamp) as day
from default.churn_submits
    ), user_days as (
select
    user_id, day
from users
    cross join days
    ), activity as (
select
    user_id, toDate(timestamp) as day, count (*) as n_submits, count (distinct task_id) as n_tasks, sum (is_solved) as n_solved
from default.churn_submits
group by user_id, day
    )
select ud.user_id,
       ud.day,
       coalesce(a.n_submits, 0) as n_submits,
       coalesce(a.n_tasks, 0)   as n_tasks,
       coalesce(a.n_solved, 0)  as n_solved
from user_days ud
         left join activity a
                   on ud.user_id = a.user_id and ud.day = a.day
order by ud.user_id, ud.day
