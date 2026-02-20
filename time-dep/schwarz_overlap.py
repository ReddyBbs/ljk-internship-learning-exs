import numpy as np
import scipy as sp

def A_mat(dx: float, N: int, nu: float = 1., a: float = 1., c: float = .1):
    a0 = nu/dx**2 + c/2
    a1 = -nu/(2*dx**2) + a/(4*dx)
    am1 = -nu/(2*dx**2) - a/(4*dx)

    diag0 = np.concatenate(
            [
                [0],
                *[a0]*(N-1),
                [0],
            ]
            )

    diag1 = np.concatenate(
            [
                [0],
                *[a1]*(N-1),
            ]
            )

    diagm1 = np.concatenate(
            [
                *[am1]*(N-1),
                [0],
            ]
            )

    A = sp.sparse.diags_array([diag0, diag1, diagm1], offsets=[0,1,-1])

    return A.tocsr()

def time_stepper(Un, A, dt, N):
    In = sp.sparse.eye_array(N+1).tocsr()

    Unp1 = sp.linalg.solve()
