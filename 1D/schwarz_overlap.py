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


def err_inf(a, b):
    return np.max(np.abs(a - b))


def main():
    image_dir = "graphs/with_overlap"
    if not os.path.isdir(image_dir):
        os.makedirs(image_dir)

    a, b = 0, 1
    u_a = 0
    u_b = np.sin(3)
    alpha = 1

    def f(x):
        return (9 + alpha) * np.sin(3 * x)

    N = 500
    h = (b - a) / N
    N1 = 255
    N2 = 255
    x1, x2 = a + h * N1, b - h * N2
    x1_arr = np.linspace(a, x1, N1 + 1, endpoint=True)
    x2_arr = np.linspace(x2, b, N2 + 1, endpoint=True)

    nb_iter = 51

    u_1_mult = np.zeros((nb_iter, N1 + 1))
    u_2_mult = np.zeros((nb_iter, N2 + 1))

    u_1_x2 = 0
    u_2_x1 = 0

    for i in range(nb_iter):
        A_1 = A_h(alpha, h, N1)
        F_1 = RHS(u_a, u_2_x1, x1_arr, f)
        u_1_mult[i] = np.linalg.solve(A_1, F_1)
        u_1_x2 = u_1_mult[i, N - N2]

        A_2 = A_h(alpha, h, N2)
        F_2 = RHS(u_1_x2, u_b, x2_arr, f)
        u_2_mult[i] = np.linalg.solve(A_2, F_2)
        u_2_x1 = u_2_mult[i, N1 - N + N2]

    u_1_add = np.zeros((nb_iter, N1 + 1))
    u_2_add = np.zeros((nb_iter, N2 + 1))

    u_1_x2 = 0
    u_2_x1 = 0

    for i in range(nb_iter):
        A_1 = A_h(alpha, h, N1)
        F_1 = RHS(u_a, u_2_x1, x1_arr, f)
        u_1_add[i] = np.linalg.solve(A_1, F_1)

        A_2 = A_h(alpha, h, N2)
        F_2 = RHS(u_1_x2, u_b, x2_arr, f)
        u_2_add[i] = np.linalg.solve(A_2, F_2)

        u_1_x2 = u_1_add[i, N - N2]
        u_2_x1 = u_2_add[i, N1 - N + N2]

    ### Graphing of the approximations
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 9.6))

    fig.suptitle(
        "Successive iterations of the multiplicative method"
        + f"\n $x_1 = {x1} , x_2 = {x2}$\n"
        + f"$N_1 = {N1} ; N_2 = {N2}$ ; {N1 + N2 - N} overlap points"
    )

    axes[0].set_xlabel("$x$")

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
    for i in range(0, nb_iter, 10):
        axes[0].plot(x1_arr, u_1_add[i], ls="--", marker="+", label=f"$u_1^{{{i+1}}}$")
        axes[1].plot(x2_arr, u_2_add[i], ls="--", marker="+", label=f"$u_2^{{{i+1}}}$")

    axes[0].legend()
    axes[1].legend()
    axes[0].grid()
    axes[1].grid()

    fig.savefig(f"{image_dir}/multiplicative_method_simulation")

    ### General stopping criteria: stabilisation of the approximation
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 9.6), sharex=True)

    fig.suptitle(
        r"$\| u_i^k - u_i^{k-1} \|_\infty$"
        + f"\n $x_1 = {x1} , x_2 = {x2}$\n"
        + f"$N_1 = {N1} ; N_2 = {N2}$ ; {N1 + N2 - N} overlap points"
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

    axes[1].set_xticks(range(0, nb_iter, 5))

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
        + f"\n $x_1 = {x1} , x_2 = {x2}$\n"
        + f"$N_1 = {N1} ; N_2 = {N2}$ ; {N1 + N2 - N} overlap points"
    )

    u_1_ex = u_exact(x1_arr)
    u_2_ex = u_exact(x2_arr)

    axes[1].set_xlabel("Iteration")
    axes[0].set_title(r"$i = 1$")
    axes[1].set_title(r"$i = 2$")

    axes[0].plot(
        range(nb_iter),
        [err_inf(u_1_add[i], u_1_ex) for i in range(nb_iter)],
        ls="--",
        marker="+",
        label="Additive method",
    )
    axes[1].plot(
        range(nb_iter),
        [err_inf(u_2_add[i], u_2_ex) for i in range(nb_iter)],
        ls="--",
        marker="+",
        label="Additive method",
    )
    axes[0].plot(
        range(nb_iter),
        [err_inf(u_1_mult[i], u_1_ex) for i in range(nb_iter)],
        ls="--",
        marker="+",
        label="Multiplicative method",
    )
    axes[1].plot(
        range(nb_iter),
        [err_inf(u_2_mult[i], u_2_ex) for i in range(nb_iter)],
        ls="--",
        marker="+",
        label="Multiplicative method",
    )

    axes[1].set_xticks(range(0, nb_iter, 5))

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
