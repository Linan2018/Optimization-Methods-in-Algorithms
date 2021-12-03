from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib import cm
from matplotlib.font_manager import FontProperties
from sympy.abc import x, y

from simplex import Simplex

font = FontProperties(fname="font/SimHei.ttf")


def evalf(f, subs):
    return np.array(f.evalf(subs=subs)).astype(np.float64)


def penalty(g, subs, beta=1e9):
    return beta * np.sum(np.max(np.hstack([evalf(g, subs), np.zeros((len(g), 1))]), axis=1))


def get_lagrange_function(f, g, var_l):
    return f + g.dot(np.array(var_l).reshape((-1, 1)))


def get_subs(array, variables):
    return {str(var): value for var, value in zip(variables, array.flatten())}


def get_lagrange_approximation(lagrange, x_k, l_k, var_x, var_l):
    subs_x = get_subs(x_k, var_x)
    subs_l = get_subs(l_k, var_l)
    lagrange_v = evalf(lagrange, subs_x | subs_l)

    lagrange_diff = sp.Matrix([sp.diff(lagrange, var) for var in [x, y]])
    lagrange_diff_v = evalf(lagrange_diff, subs_x | subs_l)

    d = np.array(var_x).reshape((len(var_x), 1)) - x_k

    h = sp.hessian(lagrange, var_x)
    h_v = evalf(h, subs_x | subs_l)
    return lagrange_v + lagrange_diff_v.T.dot(d) + 0.5 * d.T.dot(h_v).dot(d)


def get_sub_problem_target(f, lagrange, x_k, l_k, var_x, var_l, var_d):
    subs_x, subs_l = get_subs(x_k, var_x), get_subs(l_k, var_l)
    f_diff = sp.Matrix([sp.diff(f, var) for var in var_x])
    f_diff_v = evalf(f_diff, subs_x)
    d = np.array(var_d).reshape((len(var_d), 1))
    h = sp.hessian(lagrange, var_x)
    h_v = evalf(h, subs_x | subs_l)
    return (f_diff_v.T.dot(d) + 0.5 * d.T.dot(h_v).dot(d))[0][0]


def get_sub_problem_constrain(g, x_k, var_x, var_d):
    subs_x = get_subs(x_k, var_x)
    g_v = evalf(g, subs_x)
    g_diff_t = sp.Matrix([[sp.diff(g[i], var) for var in var_x] for i in range(len(g))])
    g_diff_t_v = evalf(g_diff_t, subs_x)
    d_ = np.array(var_d).reshape((len(var_d), 1))
    return sp.Matrix(g_v + g_diff_t_v.dot(d_))


def get_qp_H_c(qp_target, var_d):
    coff = qp_target.as_coefficients_dict()
    H = np.zeros((len(var_d), len(var_d)))
    c = np.zeros((len(var_d), 1))
    for i, d in enumerate(var_d):
        H[i][i] = coff[d ** 2]
        c[i][0] = coff[d]
    return H, c


def get_qp_A_b(qp_constrain, var_d):
    coff = [m.as_coefficients_dict() for m in qp_constrain]
    A = np.zeros((len(qp_constrain), len(var_d)))
    b = np.zeros((len(qp_constrain), 1))
    for i in range(len(qp_constrain)):
        for j, d in enumerate(var_d):
            A[i][j] = coff[i][d]
        b[i][0] = -1 * coff[i][1]
    return A, b


def get_simplex_A_b_c(H, A, b, c):
    m = len(A)
    n = A.shape[1] + m
    H_ = np.zeros((n, n))
    H_[:H.shape[0], :H.shape[1]] = H
    A_ = np.hstack([A, np.eye(m, m)])
    A_return = np.vstack([np.hstack([A_, np.zeros((m, m + n)), np.eye(m, m), np.zeros((m, n))]),
                          np.hstack([H_, A_.T, -1 * np.eye(n, n), np.zeros((n, m)), np.eye(n, n)])])
    b_return = np.vstack([b, -1 * c, np.zeros((m, 1))])
    c_return = np.zeros((A_return.shape[1], 1))
    c_return[-1 * (m + m + n):, :] = 1
    return A_return, b_return, c_return


# def newton_method_direction(current_values, variables):
#     subs = get_subs(current_values, variables)
#     gradient_ = evalf(gradient, subs)
#     h_ = evalf(h, subs)
#     # print(gradient_, h_, subs)
#     # print(h_)
#     d_k = -1 * np.linalg.inv(h_).dot(gradient_)

def golden_section_line_search(f, g, x_k, d_k, var_x, start, end, max_steps=10):
    """黄金分割搜索"""

    def f_inner(s):
        x_new = x_k + s * d_k
        subs_x = get_subs(x_new, var_x)
        return evalf(f, subs_x) + penalty(g, subs_x)

    l = start
    r = end
    rate = 0.618
    tmp = f_inner(l)
    for i in range(max_steps):
        a = rate * l + (1 - rate) * r
        b = (1 - rate) * l + rate * r
        if f_inner(a) > f_inner(b):
            l = a
        else:
            r = b
    return l*0.5, tmp


def simple_line_search(f, g, x_k, d_k, var_x, start, end, step_size=0.01):
    def f_inner(s):
        # print(x_k)
        # print(type(d_k))
        x_new = x_k + s * d_k
        subs_x = get_subs(x_new, var_x)
        # print(f, x_new, subs_x)
        return evalf(f, subs_x) + penalty(g, subs_x)

    l = start
    r = end
    # rate = 0.618
    tmp = f_inner(l)
    best_so_far = float('inf')
    s = 0
    for i in np.arange(l, r + step_size, step_size):
        v = f_inner(i)
        if v < best_so_far:
            best_so_far = v
            s = i
    # print("@@@@",x_k, d_k, (l+r)/2, x_k + (l+r)/2 * d_k)
    return s, tmp


def sqp(f, g, x_0, lambda_0, var_x, var_lambda, var_d, max_iterations=10):
    history = defaultdict(list)
    lagrange_function = get_lagrange_function(f, g, l)
    x_k = x_0
    lambda_k = lambda_0
    search_l = 0
    search_r = 1
    for i in range(max_iterations):
        target = get_sub_problem_target(f, lagrange_function, x_k, lambda_k, var_x, var_lambda, var_d)
        constrain = get_sub_problem_constrain(g, x_k, var_x, var_d)
        print("target, constrain", target, constrain)
        H, c = get_qp_H_c(target, var_d)
        A, b = get_qp_A_b(constrain, var_d)
        # print(H, c)
        # print(A, b)
        A_simplex, b_simplex, c_simplex = get_simplex_A_b_c(H, A, b, c)
        # print(A, b, c)
        simplex_ = Simplex(c_simplex, A_simplex, b_simplex)
        try:
            best_d, _ = simplex_.run()
        except Exception:
            break
        d_k = np.array(best_d[:len(var_x)]).reshape((-1, 1))
        lambda_star = np.array(best_d[len(var_x) + len(g):len(var_x) + len(g) + len(var_lambda)]).reshape((-1, 1))

        # print(d_k, lambda_star)

        s, v = golden_section_line_search(f, g, x_k, d_k, var_x, search_l, search_r, max_steps=100)
        # s, v = simple_line_search(f, g, x_k, d_k, var_x, search_l, search_r, step_size=0.01)

        # print(s, v)
        history["x"].append(x_k.flatten().tolist())
        history["lambda"].append(lambda_k.flatten().tolist())
        history["d"].append(d_k.flatten().tolist())
        history["f(x)"].append(v)
        history["s"].append(s)

        print("d", d_k)
        print('x', x_k)
        print('s', s)
        x_k = (x_k + s * d_k).copy()
        print('after', x_k)
        lambda_k = (lambda_k + s * (lambda_star - lambda_k)).copy()
    return history


def plot_result(history, f, filename="test"):
    print("his", history)
    X = np.arange(0, 6.1, 0.1)
    Y = np.arange(0, 6.1, 0.1)
    X, Y = np.meshgrid(X, Y)
    Z = f(X, Y)

    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224)

    x1 = [x_[0] for x_ in history["x"]]
    x2 = [x_[1] for x_ in history["x"]]
    print("x1", x1)
    ax1.plot(range(len(x1)), x1, label="x1")
    ax1.plot(range(len(x2)), x2, label="x1")
    ax1.legend()

    ax2.plot(range(len(x1)), history["f(x)"])

    ax3.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.rainbow)
    ax3.view_init(elev=30,  # 仰角
                  azim=10  # 方位角
                  )
    for i, point in enumerate(history['x'][:-1]):
        print(point)
        ax4.plot([point[0], history['x'][i + 1][0]], [point[1], history['x'][i + 1][1]], c='k', marker='x')

    ax4.contourf(X, Y, Z, 100, cmap=cm.rainbow)
    CS = ax4.contour(X, Y, Z, 20)
    ax4.clabel(CS, inline=True, fontsize=12)
    ax4.scatter(9 / 4, 2, c='k', marker='x', s=200)
    a = np.arange(0, 6.01, 0.01)
    b = a ** 2
    # ax4.plot(a, b)
    ll = list(zip(a, b))

    pgon1 = plt.Polygon(ll[:ll.index((2, 4)) + 1] + [(0, 6), (6, 6), (6, 0)], closed=False, facecolor='w', alpha=0.5)

    # pgon1 = plt.Polygon(list(zip(a, b))+[(6,6), (6,0)], closed=False, facecolor='w', alpha=0.6)
    # pgon2 = plt.Polygon([(0,6), (2.001,4), (np.sqrt(6)+0.001,6)], closed=False, facecolor='w', alpha=0.6)
    ax4.add_patch(pgon1)
    # ax4.add_patch(pgon2)

    ax4.set_xlim((0, 6))
    ax4.set_ylim((0, 6))
    ax1.set_title("x1和x2值迭代变化", fontproperties=font, fontsize=18)
    ax2.set_title("f(x)值迭代变化", fontproperties=font, fontsize=18)
    ax3.set_title("函数surf", fontproperties=font, fontsize=18)
    ax4.set_title("迭代过程", fontproperties=font, fontsize=18)

    plt.savefig(f"{filename}.svg")
    plt.show()


if __name__ == '__main__':
    variables = [x, y]
    n = 2

    ff = (x - 9 / 4) ** 2 + (y - 2) ** 2


    d = [sp.var(f"d{i}") for i in range(n)]

    # l = sp.Matrix([sp.var(f"l{i}") for i in range(n)]).reshape((n, 1))
    g = sp.Matrix([[x ** 2 - y], [x + y - 6], [-x], [-y]])
    l = [sp.var(f"l{i}") for i in range(len(g))]
    # lagrange = ff + l.T.dot(g)[0][0]

    start_default = np.zeros((len(variables), 1))
    start_default2 = np.zeros((len(g), 1))
    # max_iterations = 31
    history = {}
    history["sqp"] = sqp(ff, g, start_default, start_default2, variables, l, d, max_iterations=17)

    print(history["sqp"])
    print('x', history["sqp"]["x"])
    print('f', history["sqp"]["f(x)"])
    print('d', history["sqp"]["d"])
    print('s', history["sqp"]["s"])


    def f(x, y):
        return (x - 9 / 4) ** 2 + (y - 2) ** 2


    plot_result(history["sqp"], f, filename="sqp_v2")
    for i in range(len(history["sqp"]['x'])):
        v1 = i
        v2 = history["sqp"]['x'][i]
        v3 = history["sqp"]['f(x)'][i]
        v4 = history["sqp"]['d'][i]
        v5 = history["sqp"]['s'][i]
        str_1 = v1
        str_2 = "({:.3f},{:.3f})".format(*v2)
        str_3 = "{:.3f}".format(v3)
        str_4 = "({:.3f},{:.3f})".format(*v4)
        str_5 = "{:.3f}".format(v5)
        print(f"| ${str_1}$ | ${str_2}$ | ${str_3}$ | ${str_4}$ | ${str_5}$ |")
