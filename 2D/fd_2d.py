#!/usr/bin/env python

import numpy as np
from numpy import pi
import scipy.sparse as sp
import scipy.sparse.linalg as lg
import matplotlib.pyplot as plt
from matplotlib import cm
import os


def f(xx, yy):
    R = 2 * (pi**2) * np.sin(pi * xx) * np.sin(pi * yy)
    R[0] *= 0
    R[-1] *= 0
    R[:, 0] *= 0
    R[:, -1] *= 0
    return R


def RHS(xx, yy):
    R = f(xx, yy)
    return R.flatten()


def A_h(N: int, h: float, alpha: float):
    diag_0 = np.concatenate(
        [
            np.ones(N + 1),
            *[
                np.concatenate([[1], np.ones(N - 1) * 4 / (h**2) + np.ones(N - 1) * alpha, [1]])
                for _ in range(N - 1)
            ],
            np.ones(N + 1),
        ]
    )
    diag_m1 = np.concatenate(
        [
            np.zeros(N),
            *[
                np.concatenate([[0], np.ones(N - 1) * (-1) / (h**2), [0]])
                for _ in range(N - 1)
            ],
            np.zeros(N + 1),
        ]
    )
    diag_1 = np.concatenate(
        [
            np.zeros(N + 1),
            *[
                np.concatenate([[0], np.ones(N - 1) * (-1) / (h**2), [0]])
                for _ in range(N - 1)
            ],
            np.zeros(N),
        ]
    )
    diag_N = np.concatenate(
        [
            np.zeros(N + 1),
            *[
                np.concatenate([[0], np.ones(N - 1) * (-1) / (h**2), [0]])
                for _ in range(N - 1)
            ],
        ]
    )
    diag_mN = np.concatenate(
        [
            *[
                np.concatenate([[0], np.ones(N - 1) * (-1) / (h**2), [0]])
                for _ in range(N - 1)
            ],
            np.zeros(N + 1),
        ]
    )
    A = sp.diags_array(
        [diag_0, diag_1, diag_m1, diag_N, diag_mN],
        offsets=[0, 1, -1, N + 1, -(N + 1)],
    )

    return A.tocsr()


def main():
    image_dir = "graphs"
    if not os.path.isdir(image_dir):
        os.mkdir(image_dir)

    N = 500
    h = 1 / N
    alpha = 0

    x = np.linspace(0, 1, N + 1)
    y = np.linspace(0, 1, N + 1)
    xx, yy = np.meshgrid(x, y)

    F = RHS(xx, yy)
    A = A_h(N, h, alpha)

    u = lg.spsolve(A, F)

    u = np.reshape(u, (N + 1, N + 1))

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(xx, yy, u, cmap=cm.Blues)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.suptitle(f"${N = }$")

    fig.savefig(f"{image_dir}/fd_approx_whole")


if __name__ == "__main__":
    main()
