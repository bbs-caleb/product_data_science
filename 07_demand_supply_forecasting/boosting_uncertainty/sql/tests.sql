with cte as (select log_id,
                    lag(log_id) over (order by log_id) as prev_log_id
             from Logs)
   , cte_2 as (select log_id,
                      prev_log_id,
                      case
                          when prev_log_id is not null and log_id - prev_log_id = 1 then 0
                          else 1
                          end as flg
               from cte)
   , cte_3 as (select log_id, prev_log_id, sum(flg) over (order by log_id) as grp
               from cte_2)

select *
from cte_3
