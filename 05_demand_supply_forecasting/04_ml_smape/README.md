# sMAPE and all other main metrics

В прогнозировании и регрессии часто используют процентные метрики (MAPE / WAPE / sMAPE).
С ними есть типичная проблема: **деление на ноль** и **взрывы ошибки на малых фактах**.

---

## Контекст: MAPE vs sMAPE
**MAPE**:
\[
MAPE = \frac{1}{n}\sum_{i=1}^{n}\left|\frac{y_i-\hat{y}_i}{y_i}\right|
\]
Проблема: если \(y_i \approx 0\), ошибка может стать огромной и доминировать.

**sMAPE** (symmetric MAPE):
\[
sMAPE = \frac{1}{n}\sum_{i=1}^{n}\frac{2|y_i-\hat{y}_i|}{|y_i|+|\hat{y}_i|}
\]
Она “симметричнее” и мягче, но всё ещё имеет **крайний случай**:
- если \(|y_i| + |\hat{y}_i| = 0\) (то есть \(y_i = 0\) и \(\hat{y}_i = 0\)),
  возникает \(0/0\).

**Конвенция в этой реализации**: для таких точек вклад = 0.

---

## Быстрый пример
```python
import numpy as np
from src.metrics import smape

y_true = np.array([0.5, 0.2, 100.0])
y_pred = np.array([50.0, 50.0, 110.0])

print(smape(y_true, y_pred))

