/*
  Расчет метрик точности (KPI) на SQL
  Пример агрегации данных по продажам/прогнозам
  Диалект: Standard SQL / PostgreSQL / ClickHouse
*/


select sku_id,

       -- mae: абсолютная ошибка в штуках
       avg(abs(sales_fact - sales_forecast))                              as mae,

       -- rmse: среднеквадратичная ошибка (чувствительна к выбросам)
       sqrt(avg(power(sales_fact - sales_forecast, 2)))                   as rmse,

       -- wape: взвешенная ошибка (идеально для ритейла)
       -- защита от деления на ноль через nullif
       sum(abs(sales_fact - sales_forecast)) / nullif(sum(sales_fact), 0) as wape,

       -- smape: симметричная ошибка
       avg(
               case
                   when (abs(sales_fact) + abs(sales_forecast)) = 0 then 0
                   else 2 * abs(sales_fact - sales_forecast) / (abs(sales_fact) + abs(sales_forecast))
                   end
       )                                                                  as smape

from demand_forecasts
group by sku_id
order by wape desc;
