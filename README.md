## Пятое занятие. Пайплайн на основе Snakemake.

В качестве проекта использовалось HW2 с курса ML - кластеринг. 

- В папке workflow/scripts лежит CLI с использованием click

- В файле Snakefile лежит набор правил для генерации данных и построения графиков

- Папки data и reports были сгенерированы запуском команды:
    ```bash
    snakemake all --cores all
    ```