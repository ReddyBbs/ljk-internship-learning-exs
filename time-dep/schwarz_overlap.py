#!/usr/bin/env python

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as lg
import matplotlib.pyplot as plt
from matplotlib import animation


def A_mat(dx: float, N: int, nu: float = 1.0, a: float = 1.0, c: float = 0.1):
    a0 = nu / dx**2 + c / 2
    a1 = -nu / (2 * dx**2) + a / (4 * dx)
    am1 = -nu / (2 * dx**2) - a / (4 * dx)

    diag0 = np.concatenate(
        [
            [0],
            [a0] * (N - 1),
            [0],
        ]
    )

    diag1 = np.concatenate(
        [
            [0],
            [a1] * (N - 1),
        ]
    )

    diagm1 = np.concatenate(
        [
            [am1] * (N - 1),
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

    Unp1 = lg.spsolve(LHS, RHS @ Un + (Fn + Fnp1) / 2)

    return Unp1


def std_solve(U, A, f, x_arr, dt, Nx, Nt):
    for n in range(Nt):
        U[n + 1] = time_stepper(
            U[n], A, f(x_arr, n * dt, 0, 0), f(x_arr, (n + 1) * dt, 0, 0), dt, Nx
        )
    return


def schwarz_stepper(U1, U2, A, f, x_arr, dt, Nx, Nov, Nt, U1toU2):
    for n in range(Nt):
        if U1toU2:
            U2[n + 1] = time_stepper(
                U2[n],
                A,
                f(x_arr, n * dt, U1[n + 1, -Nov], 0),
                f(x_arr, (n + 1) * dt, U1[n + 1, -Nov], 0),
                dt,
                Nx,
            )
        else:
            U1[n + 1] = time_stepper(
                U1[n],
                A,
                f(x_arr, n * dt, 0, U2[n + 1, Nov]),
                f(x_arr, (n + 1) * dt, 0, U2[n + 1, Nov]),
                dt,
                Nx,
            )
    return


def mult_schwarz(U1, U2, A1, A2, f, x1_arr, x2_arr, dt, Nx1, Nx2, Nov, Nt, nb_iter):
    for k in range(nb_iter):
        schwarz_stepper(U1[k + 1], U2[k], A1, f, x1_arr, dt, Nx1, Nov, Nt, False)
        schwarz_stepper(U1[k + 1], U2[k + 1], A2, f, x2_arr, dt, Nx2, Nov, Nt, True)
    return


def add_schwarz(U1, U2, A1, A2, f, x1_arr, x2_arr, dt, Nx1, Nx2, Nov, Nt, nb_iter):
    for k in range(nb_iter):
        schwarz_stepper(U1[k + 1], U2[k], A1, f, x1_arr, dt, Nx1, Nov, Nt, False)
        schwarz_stepper(U1[k], U2[k + 1], A2, f, x2_arr, dt, Nx2, Nov, Nt, True)
    return


nu = 1.0
a = 1.0
c = 0.1

T = 5.0
L = 5.0

Nx = 1000
dx = 2 * L / Nx
x_arr = np.linspace(-L, L, Nx + 1, endpoint=True)

Nov = 5

Nx1 = 510
x1 = -L + Nx1 * dx
x1_arr = np.linspace(-L, x1, Nx1 + 1, endpoint=True)

Nx2 = Nx - Nx1 + Nov
x2 = L - Nx2 * dx
x2_arr = np.linspace(x2, L, Nx2 + 1, endpoint=True)

Nt = 200
dt = T / Nt

U_whole = np.zeros((Nt + 1, Nx + 1))
U_whole[0] = np.exp(-(x_arr**2))

nb_iter = 20

U1 = np.zeros((nb_iter + 1, Nt + 1, Nx1 + 1))
U2 = np.zeros((nb_iter + 1, Nt + 1, Nx2 + 1))
for k in range(nb_iter + 1):
    U1[k, 0] = np.exp(-(x1_arr**2))
    U2[k, 0] = np.exp(-(x2_arr**2))

A = A_mat(dx, Nx, nu, a, c)
A1 = A_mat(dx, Nx1, nu, a, c)
A2 = A_mat(dx, Nx2, nu, a, c)

std_solve(U_whole, A, f, x_arr, dt, Nx, Nt)
mult_schwarz(U1, U2, A1, A2, f, x1_arr, x2_arr, dt, Nx1, Nx2, Nov, Nt, nb_iter)

# animation
# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(2)
ax = plt.axes(xlim=(-(L+.1), L+.1), ylim=(-0.1, 1.1))
ax.grid()
(line,) = ax.plot([], [])
(line1,) = ax.plot([], [], label="$u_1$")
(line2,) = ax.plot([], [], label="$u_2$")
ax.legend()


# initialization function: plot the background of each frame
def init():
    # line.set_data([], [])
    line1.set_data([], [])
    line2.set_data([], [])
    return (line,)


# animation function.  This is called sequentially
def animate(i):
    # line.set_data(x_arr, U_whole[i, :])
    line1.set_data(x1_arr, U1[-1, i, :])
    line2.set_data(x2_arr, U2[-1, i, :])
    return (line,)


# call the animator.  blit=True means only re-draw the parts that have changed.
# careful, on a Mac blit=True is a known bug => set it to False!
anim = animation.FuncAnimation(
    fig, animate, init_func=init, frames=Nt, interval=dt * 1e3, blit=False
)

plt.show()
