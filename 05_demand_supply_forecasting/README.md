# Demand & supply forecasting (retail)

Портфолио задач из ритейл-контекста: мастер-данные (SKU), качество данных, отчётность/мониторинг, управление запасами и базовые блоки ценообразования и прогнозирования.

---

## 01–08. Master Data & Data Quality 
| № | Проект | Что показывает |
|---|--------|----------------|
| 01 | [01_price_parser](./01_price_parser) | Очистка/нормализация цен: форматы, типы, подготовка к загрузке/витринам |
| 02 | [02_yaml_config_recursion](./02_yaml_config_recursion) | Конфиг-driven правила обработки|
| 03 | [03_data_drift](./03_data_drift) | Мониторинг качества: дрифт/аномалии/пропуски
| 04 | [04_ml_smape](./04_ml_smape) | Основные метрики KPI: WAPE, SMAPE, MSE, RMSE, etc |
| 05 | [05_stocks_gmv_inventory_postprocessor](./05_stocks_gmv_inventory_postprocessor) | Постобработка витрин остатков/GMV: консистентность полей и бизнес-правила |
| 06 | [06_sku_distribution](./06_sku_distribution) | Анализ распределения SKU: перекосы, 'дыры' ассортимента, кандидаты на чистку |
| 07 | [07_stock_supply](./07_stock_supply) | Управление запасами: дефицит/излишек, базовые правила пополнения |
| 08 | [08_asymmetric_metrics](./08_asymmetric_metrics) | Асимметричные штрафы (недосток vs пересток) |

---

## 09–12. Pricing & merchandising 
| № | Проект | Что показывает |
|---|--------|----------------|
| 09 | [09_competitor_price](./09_competitor_price) | Мониторинг цен конкурентов: агрегации - сводка для категорийных |
| 10 | [10_target_margin](./10_target_margin) | Расчёт цен под целевую маржинальность с ограничениями |
| 11 | [11_coca_cola_principle](./11_coca_cola_principle) | Постпроцессинг цен под бизнес-правила  |
| 12 | [12_surge_pricing](./12_surge_pricing) | Динамическое ценообразование как набор правил/ограничений |

---

## 13–18. Forecasting
| № | Проект | Что показывает |
|---|--------|----------------|
| 13 | [13_demand_forecast](./13_demand_forecast) | Базовый прогноз спроса для планирования |
| 14 | [14_elasticity_feature](./14_elasticity_feature) | Эластичность спроса: чувствительность SKU к цене |
| 15 | [15_decision_tree](./15_decision_tree) | Интерпретируемая модель (дерево правил/сегментация) |
| 16 | [16_gradient_boosting](./16_gradient_boosting) | ML-прогнозирование на бустинге |
| 17 | [17_boosting_uncertainty](./17_boosting_uncertainty) | Надёжность прогнозов (оценка неопределённости) |
| 18 | [18_temporal_fusion_transformers](./18_temporal_fusion_transformers) | Продвинутые временные ряды (TFT) |



## Стек

`Python` `SQL` `ClickHouse` `Excel/Power BI`, `базовый ML/Time-Series`, `scikit-learn` `pandas` `statsmodels` 

