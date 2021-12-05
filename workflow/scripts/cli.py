import sys
sys.path.append("/Users/ivansalenek/Documents/Учёба/ML_practices/HW3")

import click
import numpy as np
import src

@click.group()
def cli():
    pass

@cli.command()
@click.argument("n_samples", type=click.INT)
@click.argument("output_paths", type=click.Path(), nargs=2)
def generate_blobs(n_samples, output_paths):
    X_1, y_1 = src.generate_blobs(n_samples)
    np.savetxt(output_paths[0], X_1, delimiter = ",")
    np.savetxt(output_paths[1], y_1, delimiter = ",", fmt='% 4d')

@cli.command()
@click.argument("n_samples", type=click.INT)
@click.argument("output_paths", type=click.Path(), nargs=2)
def generate_moons(n_samples, output_paths):
    X_2, y_2 = src.make_moons(n_samples, noise=0.075)
    np.savetxt(output_paths[0], X_2, delimiter = ",")
    np.savetxt(output_paths[1], y_2, delimiter = ",", fmt='% 4d')

@cli.command()
@click.argument("n_clusters", type=click.INT)
@click.argument("clssifier", type=click.STRING)
@click.argument("input_paths", type=click.Path())
@click.argument("output_path", type=click.Path())
def kmeans(n_clusters, clssifier, input_paths, output_path):
    X = np.loadtxt(input_paths, delimiter = ",")
    kmeans = src.KMeans(n_clusters, clssifier)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    np.savetxt(output_path, labels, delimiter = ",", fmt='% 4d')

@cli.command()
@click.argument("input_paths", type=click.Path(), nargs=2)
@click.argument("output_path", type=click.Path(), nargs=1)
def visualize_clasters(input_paths, output_path):
    X = np.loadtxt(input_paths[0], delimiter = ",")
    labels = np.loadtxt(input_paths[1], dtype = int,delimiter = ",")
    fig = src.visualize_clasters(X, labels)
    fig.savefig(output_path)


if __name__ == "__main__":
    cli()
