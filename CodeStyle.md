# Инструменты

Для форматирования и проверки качества кода были выбраны 3 инструмета:

- форматтер **black**
- линтер **Pylint**
- анализатор **Radon**

## black

Преимуществом **black** является то, что его практически нельзя конфигурировать. Он сам навязывает стиль кода, который считает правильным. Благодаря этому не нужно тратить время на тонкую настройку форматтера.

**black** был запущен для всех скриптов без специальных ключей следующей командой: 

```
black .
```

**black** форматирует скрипты .py в текущей дирректории рекрсивно согласно стилю. 

```
reformatted ./src/Lasso.py
reformatted ./src/preprocessing.py
All done! ✨ 🍰 ✨
2 files reformatted.
```

В результате получился более читаемый код.

## pylint

В качестве линтера был взят **pylint**. Он используется для анализа логики и стилистики кода. Из-за своей строгости к написанию кода, на больших проектах он может выдавать множество ошибок, поэтому при его запуске лучше выбирать те правила, которые действительно важны. Так как данная часть проекта является незначительным по объему, **pylint** был запущен без указания конкретных правил (по умолчанию все они включены):

```
pylint Lasso.py
```


```
************* Module Lasso
Lasso.py:1:0: C0103: Module name "Lasso" doesn't conform to snake_case naming style (invalid-name)
Lasso.py:1:0: C0114: Missing module docstring (missing-module-docstring)
Lasso.py:20:8: W0621: Redefining name 'result' from outer scope (line 124) (redefined-outer-name)
Lasso.py:11:0: C0116: Missing function or method docstring (missing-function-docstring)
Lasso.py:27:0: C0103: Function name "RMSE" doesn't conform to snake_case naming style (invalid-name)
Lasso.py:27:0: C0103: Argument name "a" doesn't conform to snake_case naming style (invalid-name)
Lasso.py:27:0: C0103: Argument name "b" doesn't conform to snake_case naming style (invalid-name)
Lasso.py:27:0: C0116: Missing function or method docstring (missing-function-docstring)
Lasso.py:32:0: C0103: Function name "rolling_window2D" doesn't conform to snake_case naming style (invalid-name)
Lasso.py:32:0: C0103: Argument name "a" doesn't conform to snake_case naming style (invalid-name)
Lasso.py:32:0: C0103: Argument name "n" doesn't conform to snake_case naming style (invalid-name)
Lasso.py:32:0: C0116: Missing function or method docstring (missing-function-docstring)
Lasso.py:40:8: W0621: Redefining name 'i' from outer scope (line 108) (redefined-outer-name)
Lasso.py:39:0: C0103: Function name "FV1" doesn't conform to snake_case naming style (invalid-name)
Lasso.py:39:0: C0116: Missing function or method docstring (missing-function-docstring)
Lasso.py:103:0: C0103: Constant name "len_test" doesn't conform to UPPER_CASE naming style (invalid-name)
Lasso.py:3:0: W0611: Unused matplotlib.pyplot imported as plt (unused-import)
Lasso.py:8:0: W0611: Unused StandardScaler imported from sklearn.preprocessing (unused-import)

-----------------------------------
Your code has been rated at 6.90/10
```


```
pylint preprocessing.py
```

```
************* Module preprocessing
preprocessing.py:1:0: C0114: Missing module docstring (missing-module-docstring)
preprocessing.py:12:0: C0103: Constant name "stock_name" doesn't conform to UPPER_CASE naming style (invalid-name)
preprocessing.py:33:0: C0103: Function name "Future_to_MOEX" doesn't conform to snake_case naming style (invalid-name)
preprocessing.py:33:0: C0103: Argument name "x" doesn't conform to snake_case naming style (invalid-name)
preprocessing.py:33:0: C0116: Missing function or method docstring (missing-function-docstring)
preprocessing.py:153:14: E1136: Value 'future_Finam' is unsubscriptable (unsubscriptable-object)
preprocessing.py:7:0: W0611: Unused isfile imported from os.path (unused-import)
preprocessing.py:7:0: W0611: Unused join imported from os.path (unused-import)
preprocessing.py:8:0: W0611: Unused import copy (unused-import)
preprocessing.py:10:0: W0611: Unused matplotlib.pyplot imported as plt (unused-import)
preprocessing.py:3:0: C0411: standard import "import datetime as dt" should be placed before "import numpy as np" (wrong-import-order)
preprocessing.py:4:0: C0411: standard import "import re" should be placed before "import numpy as np" (wrong-import-order)
preprocessing.py:6:0: C0411: standard import "from os import listdir" should be placed before "import numpy as np" (wrong-import-order)
preprocessing.py:7:0: C0411: standard import "from os.path import isfile, join" should be placed before "import numpy as np" (wrong-import-order)
preprocessing.py:8:0: C0411: standard import "import copy" should be placed before "import numpy as np" (wrong-import-order)

-----------------------------------
Your code has been rated at 8.10/10
```

### Cтатисика по **pylint**
Видно, что подавляющее большинство ошибок в написании кода было связано с неправильными именами функций/переменных/аргументов/констант (invalid-name). Всего таких ошибок 12 из 33

Далее следуют ошибки, связанные с отсутствием docstring у функций/модулей (missing-module-docstring и missing-function-docstring). - 7/33

Следующими по частоте были ошибки, связанные с импортом библиотек: (unused-import и wrong-import-order). - 6/33 и 5/33 ошибок соответсвенно.

Наименьшими по частоте были ошибки с именами локальных переменных, идентичных глобальным (redefined-outer-name) - 2/33; и (unsubscriptable-object) - 1/33. 

## radon

В качестве анализатора кода я взял **radon**, чтобы посмотреть на разные метрики качества моего кода.

### Цикломатическая сложность:
```
radon cc .
```

Результат:
```
preprocessing.py
    F 33:0 Future_to_MOEX - A
Lasso.py
    F 11:0 scale - A
    F 39:0 FV1 - A
    F 27:0 RMSE - A
    F 32:0 rolling_window2D - A
```
Так как в коде исползовались простые функции, типа подсчёта RMSE и скейлера, их цикломатическая сложность ожидаемо низкая.

### Метрики Холстеда


```
radon hal .
```

Данные метрики описывают сложность программы и количество усилий, предположительно затраченное на написание и понимание кода.

```
preprocessing.py:
    h1: 8
    h2: 82
    N1: 51
    N2: 96
    vocabulary: 90
    length: 147
    calculated_length: 545.319264378683
    volume: 954.3024051604622
    difficulty: 4.682926829268292
    effort: 4468.9283363611885
    time: 248.27379646451047
    bugs: 0.31810080172015404
Lasso.py:
    h1: 5
    h2: 16
    N1: 13
    N2: 21
    vocabulary: 21
    length: 34
    calculated_length: 75.60964047443682
    volume: 149.33879237447786
    difficulty: 3.28125
    effort: 490.0179124787555
    time: 27.22321735993086
    bugs: 0.04977959745815929
```

Скрипт с препроцессингом оказался значительно сложнее в написании и понимании чем регрессия. Это вполне ожидаемо, так как в нем содержится много математических операторов и операндов для вычисления новых фичей. 

### Индекс поддерживаемости кода

Данный индекс говорит о том, насколько сложно будет поддерживать или редактировать код. Он рассчитывается на основе чисел, полученных из метрик, посчитанных выше.
```
preprocessing.py - A
Lasso.py - A
```

Индекс показал что код будет легко поддерживать и редактировать.

## Грубые ошибки

В целом ни один из инструментов не нашёл грубых ошибок в коде. Ни одну из ошибок, найденных **pylint** нельзя назвать критической. Возможно это вызвано тем, что перед применением **pylint** и **rodon** код был сперва отформатирован с помощью **black**. В противном случае могли "вылезти" более грубые ошибки, связанные с нарушением формата кода, что также делало бы его менее читаемым и понятным.