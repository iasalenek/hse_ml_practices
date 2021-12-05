import os
os.chdir("/Users/ivansalenek/Documents/Учёба/ML_practices/HW3")

from pathlib import Path

rule all:
    input:
        Path("reports/figures/random_blobs.jpg"),
        Path("reports/figures/sample_blobs.jpg"),
        Path("reports/figures/plus_blobs.jpg"),
        Path("reports/figures/random_moons.jpg"),
        Path("reports/figures/sample_moons.jpg"),
        Path("reports/figures/plus_moons.jpg")

rule generate_blobs:
    output:
        Path("data/X_1.csv"),
        Path("data/y_1.csv")
    params:
        cli=Path("workflow/scripts/cli.py"),
        n_samples = 400
    shell:
        "python {params.cli} generate-blobs {params.n_samples} {output}"

rule generate_moons:
    output:
        Path("data/X_2.csv"),
        Path("data/y_2.csv")
    params:
        cli=Path("workflow/scripts/cli.py"),
        n_samples = 400
    shell:
        "python {params.cli} generate-moons {params.n_samples} {output}"

rule generate_baseline_blobs_plt:
    input:
        Path("data/X_1.csv"),
        Path("data/y_1.csv")
    output:
        Path("reports/figures/baseline_blobs.jpg")
    params:
        cli=Path("workflow/scripts/cli.py"),
    shell:
        "python {params.cli} visualize-clasters {input} {output}"

rule kmeans_random:
    input:
        Path("data/X_1.csv")
    output:
        Path("data/predictions/kmeans_random.csv")
    params:
        cli=Path("workflow/scripts/cli.py"),
        n_clusters=4,
        clssifier="random"
    shell:
        "python {params.cli} kmeans {params.n_clusters} {params.clssifier} {input} {output}"

rule generate_random_blobs_plt:
    input:
        Path("data/X_1.csv"),
        Path("data/predictions/kmeans_random.csv")
    output:
        Path("reports/figures/random_blobs.jpg")
    params:
        cli=Path("workflow/scripts/cli.py"),
    shell:
        "python {params.cli} visualize-clasters {input} {output}"

rule kmeans_sample:
    input:
        Path("data/X_1.csv")
    output:
        Path("data/predictions/kmeans_sample.csv")
    params:
        cli=Path("workflow/scripts/cli.py"),
        n_clusters=4,
        clssifier="sample"
    shell:
        "python {params.cli} kmeans {params.n_clusters} {params.clssifier} {input} {output}"

rule generate_sample_blobs_plt:
    input:
        Path("data/X_1.csv"),
        Path("data/predictions/kmeans_sample.csv")
    output:
        Path("reports/figures/sample_blobs.jpg")
    params:
        cli=Path("workflow/scripts/cli.py"),
    shell:
        "python {params.cli} visualize-clasters {input} {output}"

rule kmeans_plus:
    input:
        Path("data/X_1.csv")
    output:
        Path("data/predictions/kmeans_plus.csv")
    params:
        cli=Path("workflow/scripts/cli.py"),
        n_clusters=4,
        clssifier="k-means++"
    shell:
        "python {params.cli} kmeans {params.n_clusters} {params.clssifier} {input} {output}"

rule generate_plus_blobs_plt:
    input:
        Path("data/X_1.csv"),
        Path("data/predictions/kmeans_plus.csv")
    output:
        Path("reports/figures/plus_blobs.jpg")
    params:
        cli=Path("workflow/scripts/cli.py"),
    shell:
        "python {params.cli} visualize-clasters {input} {output}"

rule generate_baseline_moons_plt:
    input:
        Path("data/X_2.csv"),
        Path("data/y_2.csv")
    output:
        Path("reports/figures/baseline_moons.jpg")
    params:
        cli=Path("workflow/scripts/cli.py"),
    shell:
        "python {params.cli} visualize-clasters {input} {output}"

rule kmeans_random_moons:
    input:
        Path("data/X_2.csv")
    output:
        Path("data/predictions/kmeans_random_m.csv")
    params:
        cli=Path("workflow/scripts/cli.py"),
        n_clusters=2,
        clssifier="random"
    shell:
        "python {params.cli} kmeans {params.n_clusters} {params.clssifier} {input} {output}"

rule generate_random_moons_plt:
    input:
        Path("data/X_2.csv"),
        Path("data/predictions/kmeans_random_m.csv")
    output:
        Path("reports/figures/random_moons.jpg")
    params:
        cli=Path("workflow/scripts/cli.py"),
    shell:
        "python {params.cli} visualize-clasters {input} {output}"

rule kmeans_sample_moons:
    input:
        Path("data/X_2.csv")
    output:
        Path("data/predictions/kmeans_sample_m.csv")
    params:
        cli=Path("workflow/scripts/cli.py"),
        n_clusters=2,
        clssifier="sample"
    shell:
        "python {params.cli} kmeans {params.n_clusters} {params.clssifier} {input} {output}"

rule generate_sample_moons_plt:
    input:
        Path("data/X_2.csv"),
        Path("data/predictions/kmeans_sample_m.csv")
    output:
        Path("reports/figures/sample_moons.jpg")
    params:
        cli=Path("workflow/scripts/cli.py"),
    shell:
        "python {params.cli} visualize-clasters {input} {output}"

rule kmeans_plus_moons:
    input:
        Path("data/X_2.csv")
    output:
        Path("data/predictions/kmeans_plus_m.csv")
    params:
        cli=Path("workflow/scripts/cli.py"),
        n_clusters=2,
        clssifier="k-means++"
    shell:
        "python {params.cli} kmeans {params.n_clusters} {params.clssifier} {input} {output}"

rule generate_plus_moons_plt:
    input:
        Path("data/X_2.csv"),
        Path("data/predictions/kmeans_plus_m.csv")
    output:
        Path("reports/figures/plus_moons.jpg")
    params:
        cli=Path("workflow/scripts/cli.py"),
    shell:
        "python {params.cli} visualize-clasters {input} {output}"
