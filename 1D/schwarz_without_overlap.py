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


def M1(alpha: float, lamb: float, h: float, N: int):
    A = np.zeros((N + 1, N + 1))

    for i in range(1, N):
        A[i, i - 1] = -1
        A[i, i] = 2
        A[i, i + 1] = -1

    A /= h**2

    A += alpha * np.identity(N + 1)

    A[0, 0] = 1

    A[N, N - 2] = 1 / (2 * h)
    A[N, N - 1] = -2 / h
    A[N, N] = lamb + 3 / (2 * h)

    return A


def M2(alpha: float, lamb: float, h: float, N: int):
    A = np.zeros((N + 1, N + 1))

    for i in range(1, N):
        A[i, i - 1] = -1
        A[i, i] = 2
        A[i, i + 1] = -1

    A /= h**2

    A += alpha * np.identity(N + 1)

    A[0, 0] = lamb + 3 / (2 * h)
    A[0, 1] = -2 / h
    A[0, 2] = 1 / (2 * h)

    A[N, N] = 1

    return A


def fwd_diff(lamb: float, h: float, u, i: int):
    return lamb * u[i] + (-3 * u[i] + 4 * u[i + 1] - u[i + 2]) / (2 * h)


def bckwd_diff(lamb: float, h: float, u, i: int):
    return lamb * u[i] + (-u[i - 2] + 4 * u[i - 1] - 3 * u[i]) / (2 * h)


def err_inf(a, b):
    return np.max(np.abs(a - b))


def main():
    image_dir = "graphs/without_overlap"
    if not os.path.isdir(image_dir):
        os.makedirs(image_dir)

    a, b = 0, 1
    u_a = 0
    u_b = np.sin(3)
    alpha = 1
    lamb = 1

    def f(x):
        return (9 + alpha) * np.sin(3 * x)

    N1 = 300
    N2 = 300
    x1 = 0.5
    h1 = (x1 - a) / N1
    h2 = (b - x1) / N2
    x1_arr = np.linspace(a, x1, N1 + 1, endpoint=True)
    x2_arr = np.linspace(x1, b, N2 + 1, endpoint=True)

    nb_iter = 20

    u_1_mult = np.zeros((nb_iter + 1, N1 + 1))
    u_2_mult = np.zeros((nb_iter + 1, N2 + 1))

    for i in range(nb_iter):
        A_1 = M1(alpha, lamb, h1, N1)
        F_1 = RHS(u_a, fwd_diff(lamb, h2, u_2_mult[i], 0), x1_arr, f)
        u_1_mult[i + 1] = np.linalg.solve(A_1, F_1)

        A_2 = M2(alpha, lamb, h2, N2)
        F_2 = RHS(bckwd_diff(lamb, h1, u_1_mult[i + 1], N1), u_b, x2_arr, f)
        u_2_mult[i + 1] = np.linalg.solve(A_2, F_2)

    u_1_add = np.zeros((nb_iter + 1, N1 + 1))
    u_2_add = np.zeros((nb_iter + 1, N2 + 1))

    for i in range(nb_iter):
        A_1 = M1(alpha, lamb, h1, N1)
        F_1 = RHS(u_a, fwd_diff(lamb, h2, u_2_add[i], 0), x1_arr, f)
        u_1_add[i + 1] = np.linalg.solve(A_1, F_1)

        A_2 = M2(alpha, lamb, h2, N2)
        F_2 = RHS(bckwd_diff(lamb, h1, u_1_add[i], N1), u_b, x2_arr, f)
        u_2_add[i + 1] = np.linalg.solve(A_2, F_2)

    ## Graphing of the approximations
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 9.6), sharex=True)

    fig.suptitle(
        "Successive iterations of the multiplicative method\n"
        + f"$x_1 = {x1}$\n$N_1 = {N1} ; N_2 = {N2}$"
    )

    axes[1].set_xlabel("$x$")

    axes[0].plot(
        np.linspace(a, b, 500, endpoint=True),
        u_exact(np.linspace(a, b, 500, endpoint=True)),
        ls="-",
        color="k",
        label=r"Exact solution, $u(x) = \sin (3x)$",
    )
    axes[1].plot(
        np.linspace(a, b, 500, endpoint=True),
        u_exact(np.linspace(a, b, 500, endpoint=True)),
        ls="-",
        color="k",
        label=r"Exact solution, $u(x) = \sin (3x)$",
    )
    for i in [*range(4), *range(4, nb_iter + 1, 5)]:
        axes[0].plot(x1_arr, u_1_mult[i], ls="--", marker="+", label=f"$u_1^{{{i}}}$")
        axes[1].plot(x2_arr, u_2_mult[i], ls="--", marker="+", label=f"$u_2^{{{i}}}$")

    axes[0].legend()
    axes[1].legend()
    axes[0].grid()
    axes[1].grid()

    fig.tight_layout()
    fig.savefig(f"{image_dir}/multiplicative_method_simulation")

    ### General stopping criteria: stabilisation of the approximation
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 9.6), sharex=True)

    fig.suptitle(
        r"$\| u_i^k - u_i^{k-1} \|_\infty$"
        + f"\n $x_1 = {x1}$\n"
        + f"$N_1 = {N1} ; N_2 = {N2}$\n$\\lambda = {lamb}$"
    )

    axes[1].set_xlabel("Iteration")
    axes[0].set_title(r"$i = 1$")
    axes[1].set_title(r"$i = 2$")

    axes[0].plot(
        range(1, nb_iter),
        [err_inf(u_1_add[i], u_1_add[i - 1]) for i in range(1, nb_iter)],
        ls="--",
        marker="+",
        label="Additive method",
    )
    axes[1].plot(
        range(1, nb_iter),
        [err_inf(u_2_add[i], u_2_add[i - 1]) for i in range(1, nb_iter)],
        ls="--",
        marker="+",
        label="Additive method",
    )
    axes[0].plot(
        range(1, nb_iter),
        [err_inf(u_1_mult[i], u_1_mult[i - 1]) for i in range(1, nb_iter)],
        ls="--",
        marker="+",
        label="Multiplicative method",
    )
    axes[1].plot(
        range(1, nb_iter),
        [err_inf(u_2_mult[i], u_2_mult[i - 1]) for i in range(1, nb_iter)],
        ls="--",
        marker="+",
        label="Multiplicative method",
    )

    axes[1].set_xticks(range(nb_iter))

    axes[0].set_yscale("log")
    axes[1].set_yscale("log")

    axes[0].legend()
    axes[1].legend()
    axes[0].grid()
    axes[1].grid()

    fig.tight_layout()
    fig.savefig(f"{image_dir}/stopping_criteria_comparison")

    ### Convergence graph
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 9.6), sharex=True)

    fig.suptitle(
        r"$\| u_i^k - u_{|\Omega_i} \|_\infty$"
        + f"\n $x_1 = {x1}$\n"
        + f"$N_1 = {N1} ; N_2 = {N2}$"
    )

    u_1_ex = u_exact(x1_arr)
    u_2_ex = u_exact(x2_arr)

    axes[1].set_xlabel("Iteration")
    axes[0].set_title(r"$i = 1$")
    axes[1].set_title(r"$i = 2$")

    for lamb in [4., 3.5, 3., 2.5, 2., 1.5, 1., .5]:
        u_1_mult = np.zeros((nb_iter + 1, N1 + 1))
        u_2_mult = np.zeros((nb_iter + 1, N2 + 1))

        for i in range(nb_iter):
            A_1 = M1(alpha, lamb, h1, N1)
            F_1 = RHS(u_a, fwd_diff(lamb, h2, u_2_mult[i], 0), x1_arr, f)
            u_1_mult[i + 1] = np.linalg.solve(A_1, F_1)

            A_2 = M2(alpha, lamb, h2, N2)
            F_2 = RHS(bckwd_diff(lamb, h1, u_1_mult[i + 1], N1), u_b, x2_arr, f)
            u_2_mult[i + 1] = np.linalg.solve(A_2, F_2)

        axes[0].plot(
            range(1, nb_iter+1),
            [err_inf(u_1_mult[i], u_1_ex) for i in range(1, nb_iter+1)],
            ls="--",
            marker="+",
            label=f"$\\lambda = {lamb}$",
        )
        axes[1].plot(
            range(1, nb_iter+1),
            [err_inf(u_2_mult[i], u_2_ex) for i in range(1, nb_iter+1)],
            ls="--",
            marker="+",
            label=f"$\\lambda = {lamb}$",
        )

    axes[1].set_xticks(range(1, nb_iter+1))

    axes[0].set_yscale("log")
    axes[1].set_yscale("log")

    axes[0].legend()
    axes[1].legend()
    axes[0].grid()
    axes[1].grid()

    fig.tight_layout()
    fig.savefig(f"{image_dir}/error_graph")


if __name__ == "__main__":
    main()
