with
-- базовая агрегация по неделям для каждого продукта
product_weekly as (
    select
        product_name,
        tomonday(dt) as monday,
        max(price) as max_price,
        count(*) as y
    from default.data_sales_train
    group by product_name, monday
),

-- расчёт лагов для каждого продукта
product_lags as (
    select
        product_name,
        monday,
        max_price,
        y,
        laginframe(y, 1, 0) over w as y_lag_1,
        laginframe(y, 2, 0) over w as y_lag_2,
        laginframe(y, 3, 0) over w as y_lag_3,
        laginframe(y, 4, 0) over w as y_lag_4,
        laginframe(y, 5, 0) over w as y_lag_5,
        laginframe(y, 6, 0) over w as y_lag_6
    from product_weekly
    window w as (partition by product_name order by monday rows between unbounded preceding and unbounded following)
),

-- расчёт фичей для каждого продукта (avg/max/min на основе лагов)
product_features as (
    select
        product_name,
        monday,
        max_price,
        y,
        y_lag_1,
        y_lag_2,
        y_lag_3,
        y_lag_4,
        y_lag_5,
        y_lag_6,
        -- скользящие агрегаты за 3 недели
        (y_lag_1 + y_lag_2 + y_lag_3) / 3.0 as y_avg_3,
        greatest(y_lag_1, y_lag_2, y_lag_3) as y_max_3,
        least(y_lag_1, y_lag_2, y_lag_3) as y_min_3,
        -- скользящие агрегаты за 6 недель
        (y_lag_1 + y_lag_2 + y_lag_3 + y_lag_4 + y_lag_5 + y_lag_6) / 6.0 as y_avg_6,
        greatest(y_lag_1, y_lag_2, y_lag_3, y_lag_4, y_lag_5, y_lag_6) as y_max_6,
        least(y_lag_1, y_lag_2, y_lag_3, y_lag_4, y_lag_5, y_lag_6) as y_min_6
    from product_lags
),

-- суммируем лаги по всем продуктам для каждой недели (y_all_lag_i = sum(y_lag_i))
global_lags as (
    select
        monday,
        sum(y_lag_1) as y_all_lag_1,
        sum(y_lag_2) as y_all_lag_2,
        sum(y_lag_3) as y_all_lag_3,
        sum(y_lag_4) as y_all_lag_4,
        sum(y_lag_5) as y_all_lag_5,
        sum(y_lag_6) as y_all_lag_6
    from product_lags
    group by monday
),

-- расчёт глобальных фичей на основе сумм лагов
global_features as (
    select
        monday,
        y_all_lag_1,
        y_all_lag_2,
        y_all_lag_3,
        y_all_lag_4,
        y_all_lag_5,
        y_all_lag_6,
        -- скользящие агрегаты за 3 недели
        (y_all_lag_1 + y_all_lag_2 + y_all_lag_3) / 3.0 as y_all_avg_3,
        greatest(y_all_lag_1, y_all_lag_2, y_all_lag_3) as y_all_max_3,
        least(y_all_lag_1, y_all_lag_2, y_all_lag_3) as y_all_min_3,
        -- скользящие агрегаты за 6 недель
        (y_all_lag_1 + y_all_lag_2 + y_all_lag_3 + y_all_lag_4 + y_all_lag_5 + y_all_lag_6) / 6.0 as y_all_avg_6,
        greatest(y_all_lag_1, y_all_lag_2, y_all_lag_3, y_all_lag_4, y_all_lag_5, y_all_lag_6) as y_all_max_6,
        least(y_all_lag_1, y_all_lag_2, y_all_lag_3, y_all_lag_4, y_all_lag_5, y_all_lag_6) as y_all_min_6
    from global_lags
)

-- объединение всех фичей в итоговый датасет
select
    pf.product_name,
    pf.monday,
    pf.max_price,
    pf.y,
    pf.y_lag_1,
    pf.y_lag_2,
    pf.y_lag_3,
    pf.y_lag_4,
    pf.y_lag_5,
    pf.y_lag_6,
    pf.y_avg_3,
    pf.y_max_3,
    pf.y_min_3,
    pf.y_avg_6,
    pf.y_max_6,
    pf.y_min_6,
    gf.y_all_lag_1,
    gf.y_all_lag_2,
    gf.y_all_lag_3,
    gf.y_all_lag_4,
    gf.y_all_lag_5,
    gf.y_all_lag_6,
    gf.y_all_avg_3,
    gf.y_all_max_3,
    gf.y_all_min_3,
    gf.y_all_avg_6,
    gf.y_all_max_6,
    gf.y_all_min_6
from product_features pf
left join global_features gf on pf.monday = gf.monday
order by pf.product_name, pf.monday;

