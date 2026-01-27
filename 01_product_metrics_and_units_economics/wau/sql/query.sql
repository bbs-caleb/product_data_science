with recursive
    lvl as (select employee_id, 1 as level
            from employees
            where manager_id is null

            union all

            select e.employee_id, l.level + 1
            from lvl l
                     join employees e
                          on e.manager_id = l.employee_id),
    rel as (select employee_id as manager_id, employee_id as subordinate_id
            from employees

            union all

            select r.manager_id, e.employee_id
            from rel r
                     join employees e
                          on e.manager_id = r.subordinate_id),
    agg as (select r.manager_id  as employee_id,
                   count(*) - 1  as team_size,
                   sum(e.salary) as budget
            from rel r
                     join employees e
                          on e.employee_id = r.subordinate_id
            group by r.manager_id)
select e.employee_id,
       e.employee_name,
       l.level,
       a.team_size,
       a.budget
from employees e
         join lvl l
              on l.employee_id = e.employee_id
         join agg a
              on a.employee_id = e.employee_id
order by l.level asc,
         a.budget desc,
         e.employee_name asc;
