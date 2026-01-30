#!/usr/bin/env python

import numpy as np
from numpy import pi
import scipy.sparse as sp
import scipy.sparse.linalg as lg
import matplotlib.pyplot as plt
from matplotlib import colormaps
import os


def f(xx, yy, left_boundary=None, right_boundary=None):
    R = 2 * (pi**2) * np.sin(pi * xx) * np.sin(pi * yy)
    R[0] *= 0
    R[-1] *= 0

    if left_boundary is None:
        left_boundary = np.zeros_like(xx[:, 0])
    if right_boundary is None:
        right_boundary = np.zeros_like(xx[:, 0])

    R[:, 0] = left_boundary
    R[:, -1] = right_boundary

    return R


def u_exact(xx, yy):
    R = np.sin(pi * xx) * np.sin(pi * yy)
    return R


def err_inf(a, b):
    return np.max(np.abs(a - b))


def RHS(xx, yy, left_boundary=None, right_boundary=None):
    R = f(xx, yy, left_boundary, right_boundary)
    return R.flatten()


def fwd_diff(lamb: float, h: float, u, i: int):
    return lamb * u[:, i] + (-3 * u[:, i] + 4 * u[:, i + 1] - u[:, i + 2]) / (2 * h)


def bckwd_diff(lamb: float, h: float, u, i: int):
    return lamb * u[:, i] + (-u[:, i - 2] + 4 * u[:, i - 1] - 3 * u[:, i]) / (2 * h)


def A_h(Nx: int, Ny: int, h: float, alpha: float):
    diag_0 = np.concatenate(
        [
            np.ones(Nx + 1),
            *[
                np.concatenate(
                    [[1], np.ones(Nx - 1) * 4 / (h**2) + np.ones(Nx - 1) * alpha, [1]]
                )
                for _ in range(Ny - 1)
            ],
            np.ones(Nx + 1),
        ]
    )
    diag_m1 = np.concatenate(
        [
            np.zeros(Nx),
            *[
                np.concatenate([[0], np.ones(Nx - 1) * (-1) / (h**2), [0]])
                for _ in range(Ny - 1)
            ],
            np.zeros(Nx + 1),
        ]
    )
    diag_1 = np.concatenate(
        [
            np.zeros(Nx + 1),
            *[
                np.concatenate([[0], np.ones(Nx - 1) * (-1) / (h**2), [0]])
                for _ in range(Ny - 1)
            ],
            np.zeros(Nx),
        ]
    )
    diag_N = np.concatenate(
        [
            np.zeros(Nx + 1),
            *[
                np.concatenate([[0], np.ones(Nx - 1) * (-1) / (h**2), [0]])
                for _ in range(Ny - 1)
            ],
        ]
    )
    diag_mN = np.concatenate(
        [
            *[
                np.concatenate([[0], np.ones(Nx - 1) * (-1) / (h**2), [0]])
                for _ in range(Ny - 1)
            ],
            np.zeros(Nx + 1),
        ]
    )
    A = sp.diags_array(
        [diag_0, diag_1, diag_m1, diag_N, diag_mN],
        offsets=[0, 1, -1, Nx + 1, -(Nx + 1)],
    )

    return A.tocsr()


def M1(Nx: int, Ny: int, h: float, alpha: float, lamb: float):
    diag_0 = np.concatenate(
        [
            np.ones(Nx),
            [lamb + 3 / (2 * h)],
            *[
                np.concatenate(
                    [
                        [1],
                        np.ones(Nx - 1) * 4 / (h**2) + np.ones(Nx - 1) * alpha,
                        [lamb + 3 / (2 * h)],
                    ]
                )
                for _ in range(Ny - 1)
            ],
            np.ones(Nx),
            [lamb + 3 / (2 * h)],
        ]
    )
    diag_1 = np.concatenate(
        [
            np.zeros(Nx + 1),
            *[
                np.concatenate([[0], np.ones(Nx - 1) * (-1) / (h**2), [0]])
                for _ in range(Ny - 1)
            ],
            np.zeros(Nx),
        ]
    )
    diag_m1 = np.concatenate(
        [
            np.zeros(Nx - 1),
            [-2 / h],
            *[
                np.concatenate([[0], np.ones(Nx - 1) * (-1) / (h**2), [-2 / h]])
                for _ in range(Ny - 1)
            ],
            np.zeros(Nx),
            [-2 / h],
        ]
    )
    diag_m2 = np.concatenate(
        [
            np.zeros(Nx - 2),
            [1 / (2 * h)],
            *[np.concatenate([np.zeros(Nx), [1 / (2 * h)]]) for _ in range(Ny)],
        ]
    )
    diag_N = np.concatenate(
        [
            np.zeros(Nx + 1),
            *[
                np.concatenate([[0], np.ones(Nx - 1) * (-1) / (h**2), [0]])
                for _ in range(Ny - 1)
            ],
        ]
    )
    diag_mN = np.concatenate(
        [
            *[
                np.concatenate([[0], np.ones(Nx - 1) * (-1) / (h**2), [0]])
                for _ in range(Ny - 1)
            ],
            np.zeros(Nx + 1),
        ]
    )
    A = sp.diags_array(
        [diag_0, diag_1, diag_m1, diag_m2, diag_N, diag_mN],
        offsets=[0, 1, -1, -2, Nx + 1, -(Nx + 1)],
    )

    A = A.tocsr()

    return A


def M2(Nx: int, Ny: int, h: float, alpha: float, lamb: float):
    diag_0 = np.concatenate(
        [
            [lamb + 3 / (2 * h)],
            np.ones(Nx),
            *[
                np.concatenate(
                    [
                        [lamb + 3 / (2 * h)],
                        np.ones(Nx - 1) * 4 / (h**2) + np.ones(Nx - 1) * alpha,
                        [1],
                    ]
                )
                for _ in range(Ny - 1)
            ],
            [lamb + 3 / (2 * h)],
            np.ones(Nx),
        ]
    )
    diag_1 = np.concatenate(
        [
            [-2 / h],
            np.zeros(Nx),
            *[
                np.concatenate([[-2 / h], np.ones(Nx - 1) * (-1) / (h**2), [0]])
                for _ in range(Ny - 1)
            ],
            [-2 / h],
            np.zeros(Nx - 1),
        ]
    )
    diag_m1 = np.concatenate(
        [
            np.zeros(Nx),
            *[
                np.concatenate([[0], np.ones(Nx - 1) * (-1) / (h**2), [0]])
                for _ in range(Ny - 1)
            ],
            np.zeros(Nx + 1),
        ]
    )
    diag_2 = np.concatenate(
        [
            *[np.concatenate([[1 / (2 * h)], np.zeros(Nx)]) for _ in range(Ny)],
            [1 / (2 * h)],
            np.zeros(Nx - 2),
        ]
    )
    diag_N = np.concatenate(
        [
            np.zeros(Nx + 1),
            *[
                np.concatenate([[0], np.ones(Nx - 1) * (-1) / (h**2), [0]])
                for _ in range(Ny - 1)
            ],
        ]
    )
    diag_mN = np.concatenate(
        [
            *[
                np.concatenate([[0], np.ones(Nx - 1) * (-1) / (h**2), [0]])
                for _ in range(Ny - 1)
            ],
            np.zeros(Nx + 1),
        ]
    )
    A = sp.diags_array(
        [diag_0, diag_1, diag_m1, diag_2, diag_N, diag_mN],
        offsets=[0, 1, -1, 2, Nx + 1, -(Nx + 1)],
    )

    return A.tocsr()


def main():
    image_dir = "graphs"
    if not os.path.isdir(image_dir):
        os.mkdir(image_dir)

    N = 400
    N1 = 250
    N_ov = 10
    N2 = N - N1 + N_ov
    h = 1 / N
    x1 = h * N1
    x2 = 1 - h * N2
    alpha = 0

    nb_iter = 10
    lamb = 2.0

    x = np.linspace(0, 1, N + 1, endpoint=True)
    y = np.linspace(0, 1, N + 1, endpoint=True)
    x1_arr = np.linspace(0, x1, N1 + 1, endpoint=True)
    x2_arr = np.linspace(x2, 1, N2 + 1, endpoint=True)
    xx, yy = np.meshgrid(x, y)
    xx1, yy1 = np.meshgrid(x1_arr, y)
    xx2, yy2 = np.meshgrid(x2_arr, y)

    A1 = M1(N1, N, h, alpha, lamb)
    A2 = M2(N2, N, h, alpha, lamb)

    u1_mult = np.zeros((nb_iter + 1, N + 1, N1 + 1))
    u2_mult = np.zeros((nb_iter + 1, N + 1, N2 + 1))

    for i in range(nb_iter):
        F1 = RHS(xx1, yy1, right_boundary=fwd_diff(lamb, h, u2_mult[i], N_ov))
        u1_mult[i + 1] = np.reshape(lg.spsolve(A1, F1), (N + 1, N1 + 1))

        F2 = RHS(xx2, yy2, left_boundary=bckwd_diff(lamb, h, u1_mult[i + 1], N1 - N_ov))
        u2_mult[i + 1] = np.reshape(lg.spsolve(A2, F2), (N + 1, N2 + 1))

    u1_add = np.zeros((nb_iter + 1, N + 1, N1 + 1))
    u2_add = np.zeros((nb_iter + 1, N + 1, N2 + 1))

    for i in range(nb_iter):
        F1 = RHS(xx1, yy1, right_boundary=fwd_diff(lamb, h, u2_add[i], N_ov))
        u1_add[i + 1] = np.reshape(lg.spsolve(A1, F1), (N + 1, N1 + 1))

        F2 = RHS(xx2, yy2, left_boundary=bckwd_diff(lamb, h, u1_add[i], N1 - N_ov))
        u2_add[i + 1] = np.reshape(lg.spsolve(A2, F2), (N + 1, N2 + 1))

    fig = plt.figure(figsize=(6.4, 2 * 4.8))
    fig.suptitle(
        f"\n $x_1 = {x1} , x_2 = {x2}$\n"
        + f"$N_1 = {N1} ; N_2 = {N2}$ ; {N_ov} overlap points"
    )

    ax1 = fig.add_subplot(2, 1, 1, projection="3d")
    ax2 = fig.add_subplot(2, 1, 2, projection="3d")

    ax1.set_title("Multiplicative method")
    ax1.plot_surface(xx1, yy1, u1_mult[-1], cmap=colormaps["Blues"])
    ax1.plot_surface(xx2, yy2, u2_mult[-1], cmap=colormaps["Greens"])

    ax2.set_title("Additive method")
    ax2.plot_surface(xx1, yy1, u1_add[-1], cmap=colormaps["Blues"])
    ax2.plot_surface(xx2, yy2, u2_add[-1], cmap=colormaps["Greens"])

    fig.tight_layout()
    fig.savefig(f"{image_dir}/without_overlap_graphs")

    fig = plt.figure(figsize=(6.4, 2 * 4.8))
    fig.suptitle(
        f"\n $x_1 = {x1} , x_2 = {x2}$\n"
        + f"$N_1 = {N1} ; N_2 = {N2}$ ; {N_ov} overlap points"
    )

    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax1.set_title("Multiplicative method")
    ax1.set_yscale("log")
    ax2.set_title("Additive method")
    ax2.set_yscale("log")

    for lamb in [4.0, 3.0, 2.0, 1.0, 0.5]:
        A1 = M1(N1, N, h, alpha, lamb)
        A2 = M2(N2, N, h, alpha, lamb)

        u1_mult = np.zeros((nb_iter + 1, N + 1, N1 + 1))
        u2_mult = np.zeros((nb_iter + 1, N + 1, N2 + 1))

        u1_add = np.zeros((nb_iter + 1, N + 1, N1 + 1))
        u2_add = np.zeros((nb_iter + 1, N + 1, N2 + 1))

        for i in range(nb_iter):
            F1 = RHS(xx1, yy1, right_boundary=fwd_diff(lamb, h, u2_mult[i], N_ov))
            u1_mult[i + 1] = np.reshape(lg.spsolve(A1, F1), (N + 1, N1 + 1))

            F2 = RHS(xx2, yy2, left_boundary=bckwd_diff(lamb, h, u1_mult[i + 1], N1 - N_ov))
            u2_mult[i + 1] = np.reshape(lg.spsolve(A2, F2), (N + 1, N2 + 1))

        for i in range(nb_iter):
            F1 = RHS(xx1, yy1, right_boundary=fwd_diff(lamb, h, u2_add[i], N_ov))
            u1_add[i + 1] = np.reshape(lg.spsolve(A1, F1), (N + 1, N1 + 1))

            F2 = RHS(xx2, yy2, left_boundary=bckwd_diff(lamb, h, u1_add[i], N1 - N_ov))
            u2_add[i + 1] = np.reshape(lg.spsolve(A2, F2), (N + 1, N2 + 1))

        ax1.plot(
            range(nb_iter + 1),
            [err_inf(u1_mult[i], u_exact(xx1, yy1)) for i in range(nb_iter + 1)],
            label=f"$\\lambda = {lamb}$",
            marker="+",
        )

        ax2.plot(
            range(nb_iter + 1),
            [err_inf(u1_add[i], u_exact(xx1, yy1)) for i in range(nb_iter + 1)],
            label=f"$\\lambda = {lamb}$",
            marker="+",
        )

    ax1.set_xticks(range(nb_iter + 1))
    ax2.set_xticks(range(nb_iter + 1))
    ax2.set_xlabel("Iteration")
    ax1.legend()
    ax2.legend()
    ax1.grid()
    ax2.grid()

    fig.tight_layout()
    fig.savefig(f"{image_dir}/without_overlap_errors")



if __name__ == "__main__":
    main()
