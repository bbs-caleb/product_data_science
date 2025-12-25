select
        age,
        income,
        dependents,
        has_property,
        has_car,
        credit_score,
        job_tenure,
        has_education,
        loan_amount,
        dateDiff('day', loan_start, loan_deadline) as loan_period,
        case
            when loan_payed <= loan_deadline then 0
            else dateDiff('day', loan_deadline, loan_payed)
        end as delay_days
from default.loan_delay_days
order by id

