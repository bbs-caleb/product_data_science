select
    user_id,
    toDate(timestamp) as day,
    count(*) as n_submits,
    count(distinct task_id) as n_tasks,
    sum(is_solved) as n_solved
from default.churn_submits
group by user_id, day
order by user_id, day
