#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os


def K(r: float, x0: float, alpha: float):
    beta = np.sqrt(alpha)
    x1 = x0 + r / 2
    x2 = x0 - r / 2

    K1 = (np.exp(beta * x1) - np.exp(beta * (2 - x1))) / (
        np.exp(beta * x1) - np.exp(-beta * x1)
    )
    K2 = (np.exp(beta * x2) - np.exp(-beta * x2)) / (
        np.exp(beta * x2) - np.exp(beta * (2 - x2))
    )

    return K1 * K2


def main():
    image_dir = "graphs/with_overlap"
    if not os.path.isdir(image_dir):
        os.makedirs(image_dir)

    r = np.linspace(0, 1, 500)
    plt.plot(r, K(r, .5, 1))
    plt.xlabel("Overlap size")
    plt.ylabel("$K$")
    plt.grid()
    plt.savefig(f"{image_dir}/theoretical_cvgence")


if __name__ == "__main__":
    main()
