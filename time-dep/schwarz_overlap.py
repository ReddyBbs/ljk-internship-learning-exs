import numpy as np
import scipy.sparse as sp
import scipy.linalg as lg


def A_mat(dx: float, N: int, nu: float = 1.0, a: float = 1.0, c: float = 0.1):
    a0 = nu / dx**2 + c / 2
    a1 = -nu / (2 * dx**2) + a / (4 * dx)
    am1 = -nu / (2 * dx**2) - a / (4 * dx)

    diag0 = np.concatenate(
        [
            [0],
            *[a0] * (N - 1),
            [0],
        ]
    )

    diag1 = np.concatenate(
        [
            [0],
            *[a1] * (N - 1),
        ]
    )

    diagm1 = np.concatenate(
        [
            *[am1] * (N - 1),
            [0],
        ]
    )

    A = sp.diags_array([diag0, diag1, diagm1], offsets=[0, 1, -1])

    return A.tocsr()


def f(x, t, left, right):
    F = 0 * x
    F[0] = left
    F[-1] = right
    return F


def time_stepper(Un, A, Fn, Fnp1, dt, Nx):
    In = sp.eye_array(Nx + 1).tocsr()

    LHS = A + In / dt
    LHS[0, 0] = 1
    LHS[-1, -1] = 1

    RHS = In / dt - A
    RHS[0, 0] = 0
    RHS[-1, -1] = 0

    Unp1 = lg.solve(LHS, RHS @ Un + (Fn + Fnp1) / 2)

    return Unp1


def schwarz_stepper(U1, U2, A, f, x, dt, Nx, Nov, Nt, U1toU2):
    for n in range(Nt):
        if U1toU2:
            U2[n + 1] = time_stepper(
                U2[n],
                A,
                f(x, n * dt, U1[n + 1, -Nov], 0),
                f(x, (n + 1) * dt, U1[n + 1, -Nov], 0),
                dt,
                Nx,
            )
        else:
            U1[n + 1] = time_stepper(
                U1[n],
                A,
                f(x, n * dt, 0, U2[n + 1, -Nov]),
                f(x, (n + 1) * dt, 0, U2[n + 1, -Nov]),
                dt,
                Nx,
            )
    return
