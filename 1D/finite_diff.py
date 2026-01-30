#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os


def u_exact(x):
    return np.sin(3 * x)


def RHS(u_a: float, u_b: float, x, f):
    F = f(x)
    F[0] = u_a
    F[-1] = u_b
    return F


def A_h(alpha: float, h: float, N: int):
    A = np.zeros((N + 1, N + 1))

    for i in range(1, N):
        A[i, i - 1] = -1
        A[i, i] = 2
        A[i, i + 1] = -1

    A /= h**2

    A += alpha * np.identity(N + 1)

    A[0, 0] = 1
    A[N, N] = 1

    return A


def main():
    image_dir = "graphs"
    if not os.path.isdir(image_dir):
        os.mkdir(image_dir)

    a, b = 0, 1
    u_a = 0
    u_b = np.sin(3)
    alpha = 1

    def f(x):
        return (9 + alpha) * np.sin(3 * x)

    N = 50
    h = (b - a) / N
    x = np.linspace(a, b, N + 1, endpoint=True)

    A = A_h(alpha, h, N)
    F = RHS(u_a, u_b, x, f)
    u_num = np.linalg.solve(A, F)

    fig, ax = plt.subplots(1, 1)

    ax.set_xlabel("$x$")

    ax.plot(
        np.linspace(a, b, 500, endpoint=True),
        u_exact(np.linspace(a, b, 500, endpoint=True)),
        ls="-",
        color="k",
        label=r"Exact solution, $u(x) = \sin (3x)$",
    )
    ax.plot(
        x, u_num, ls="--", marker="+", color="r", label=f"FD approximation, ${N = }$"
    )
    ax.legend()
    ax.grid()

    fig.savefig(f"{image_dir}/FD_simulation")


if __name__ == "__main__":
    main()
