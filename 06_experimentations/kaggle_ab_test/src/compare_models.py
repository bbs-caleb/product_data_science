2. compare_models — t-тест для связанных выборок
pythonfrom scipy.stats import ttest_rel

_, p_value = ttest_rel(model_scores, baseline_scores)

if p_value < alpha:
    effect_sign = 1 if avg_score > baseline_avg else -1
else:
    effect_sign = 0  # не отличается статистически



Примеры вызова
python
compare_models(cv=5, model=model, ...)

# RepeatedKFold: 5 фолдов × 3 повторения = 15 итераций
compare_models(cv=(5, 3), model=model, ...)


