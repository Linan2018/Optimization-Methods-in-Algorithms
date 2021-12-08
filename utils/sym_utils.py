import numpy as np
from sympy.core import symbols
from sympy.abc import x, y, z
import sympy as sp

from utils import *

x1, x2, x3, x4, x5, x6 = symbols('x1, x2, x3, x4, x5, x6')


def _evalf(f, subs):
    return np.array(f.evalf(subs=subs)).astype(np.float64)


def _get_subs(array, variables):
    return {str(var): value for var, value in zip(variables, array.flatten())}


def sym_eval(f, values, variables):
    if type(values) != np.ndarray:
        values = np.array(values)
    return _evalf(f, _get_subs(values, variables))


def sym_diff(f, variables):
    return sp.Matrix([sp.diff(f, var) for var in variables])


def sym_hessian(f, variables):
    return sp.hessian(f, variables)

if __name__ == '__main__':
    variables = [x, y]
    f = (x - 3) ** 4 + (x - 3 * y) ** 2

    variables = [x1, x2, x3]
    f = x1**2-4*x1*x2+x2**2+8*x3**2-x1-3*x2+x3-7

    variables = [x1, x2, x3]
    f = x1**2-x1*x2+x2**2

    print(sym_diff(f, variables))
    print(sym_hessian(f, variables))
    print()

    print("特征值")
    print(eig(np.array(sym_hessian(f, variables).tolist()).astype(np.float64))[0])

    # print(sym_eval(sym_diff(f, variables), [0, 1], variables))