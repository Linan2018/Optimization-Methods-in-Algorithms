import random
from collections import defaultdict
from pulp import *
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib import cm
from matplotlib.font_manager import FontProperties
from scipy.optimize import minimize
from sympy.abc import x, y

from simplex import Simplex

font = FontProperties(fname="SimHei.ttf")


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

def golden_section_line_search(f, g, var_x, start, end, max_steps=10):
    """黄金分割搜索"""

    def f_inner(x_value):
        subs_x = get_subs(x_value, var_x)
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

def cutting_plane(f, f_diff, g, g_diff, max_iterations=50):
    points = [(0.1, 0.1), (0.2, 0.2)]
    history = defaultdict(list)
    # points = [(random.random(), random.random()) for _ in range(2)]
    prob = LpProblem("Problem", LpMinimize)
    # 2. 建立变量
    x1 = LpVariable("x1", lowBound=0)
    x2 = LpVariable("x2", lowBound=0)
    l = LpVariable("l")

    # 3. 设置目标函数 z
    prob += l

    # 4. 施加约束
    for point in points:
        # print(f_diff(point))
        prob += f(point) + f_diff(point)[0][0]*(x1-point[0]) + f_diff(point)[1][0]*(x2-point[1]) <= l, f"constraint f {point}"
        # prob += g_diff(point)[0][0]*x1 + g_diff(point)[1][0]*x2 <= g_diff(point)[0][0]*point[0] + g_diff(point)[1][0] * point[1] - g(point), f"constraint g {point}"
    for i in range(max_iterations):
        for i_point, point in enumerate(points):
            # print(len(points))
            # print(f_diff(point))
            # prob += f(point) + f_diff(point)[0][0] * (x1 - point[0]) + f_diff(point)[1][0] * (
            #             x2 - point[1]) <= l, f"constraint f {point}"
            prob += g_diff(point)[0][0] * x1 + g_diff(point)[1][0] * x2 <= g_diff(point)[0][0] * point[0] + \
                    g_diff(point)[1][0] * point[1] - g(point), f"constraint g {i} {i_point} {point}"
        # 5. 求解
        prob.solve()
        # 8. 打印最优解的目标函数值

        # print("z= ", value(prob.objective))
        # print(points)
        history['x'].append((prob.variablesDict()['x1'].varValue, prob.variablesDict()['x2'].varValue))
        history['f(x)'].append(f((prob.variablesDict()['x1'].varValue, prob.variablesDict()['x2'].varValue)))
        history['l'].append(prob.variablesDict()['l'].varValue)
        history['u'].append(np.min([f(p) for p in points]))

        # print(history['l'])
        # print(history['u'])
        points += [(prob.variablesDict()['x1'].varValue, prob.variablesDict()['x2'].varValue)]

    # history['x'] =
    print(history)
    return history
    #
    # # 7. 打印出每个变量的最优值
    # for v in prob.variables():
    #     print(v.name, "=", v.varValue)


def ip(f, g, x_0, t_start, t_delta, m, e=1e-6, max_iterations=100):
    max_iterations = max_iterations+1
    # print("max", max_iterations)
    history = defaultdict(list)
    x_k = x_0.flatten()

    def fun(t_):
        return lambda x : t_ * f(x) - np.log(-g(x)) if g(x) <= 0 else float('inf')

    def _f(t_):
        return lambda x : f(x)

    def _phi(x):
        return - np.log(-g(x)) if g(x) <= 0 else float('inf')

    def callback(xk):
        # print(xk)
        history['x'].append(xk)
        history['psi(x)'].append(fun(xk))
    # t_start = 1
    # t_delta = 30
    t = t_start
    for i in range(max_iterations):
        # print(t)
        history["t"].append(t)
        history["x"].append(x_k.flatten().tolist())
        history["f(x)"].append(_f(t)(x_k))
        history["phi(x)"].append(_phi(x_k))
        history["psi(x)"].append(fun(t)(x_k))

        print(f'|{i}|{t}|{"({:.3f},{:.3f})".format(*(x_k.flatten().tolist()))}|{"{:.3f}".format(_f(t)(x_k))}|{"{:.3f}".format(_phi(x_k))}|{"{:.3f}".format(fun(t)(x_k))}|')

        e = abs(history["f(x)"][-1]-history["f(x)"][-2]) if len(history["f(x)"]) > 2 else -1
        # if m/t <= e:
        #     break
        # print("f_1", fun_(t)(x_k))
        # print("f_p", fun_p(x_k))

        fff = fun(t)
        # print(fff([0, 0]), fff([0.2, 0.2]))
        res = minimize(fun(t), x_k, method="Powell")
        t += t_delta
        # print(dir(res))
        x_k = res.x

    return history


def plot_result(histories, f, filename="test"):
    X = np.arange(-1, 1.1, 0.1)
    Y = np.arange(-1, 1.1, 0.1)
    X, Y = np.meshgrid(X, Y)
    def fun(x, y):
        return f([x, y])
    Z = fun(X, Y)

    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    # ax3 = fig.add_subplot(223, projection='3d')
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    x01 = [x_[0] for x_ in histories[0]["x"]]
    x02 = [x_[1] for x_ in histories[0]["x"]]
    print('x', len(x01), x01)
    ax1.plot(range(len(x01)), x01, c='r', label='(0.1, 0.1)')
    ax2.plot(range(len(x02)), x02, c='r', label='(0.1, 0.1)')

    x11 = [x_[0] for x_ in histories[1]["x"]]
    x12 = [x_[1] for x_ in histories[1]["x"]]
    ax1.plot(range(len(x11)), x11, c='b', label='(-0.1, -0.1)')
    ax2.plot(range(len(x12)), x12, c='b', label='(-0.1, -0.1)')

    ax3.plot(range(len(histories[0]['f(x)'])), histories[0]['f(x)'], c='r', label='(0.1, 0.1)')
    ax3.plot(range(len(histories[1]['f(x)'])), histories[1]['f(x)'], c='b', label='(-0.1, -0.1)')

    # ax3.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.rainbow)
    # ax3.view_init(elev=20,  # 仰角
    #               azim=10  # 方位角
    #               )

    for i, point in enumerate(histories[0]['x'][:-1]):
        # print(point)
        if i==0:
            ax4.plot([point[0], histories[0]['x'][i + 1][0]], [point[1], histories[0]['x'][i + 1][1]], c='r', label='(0.1, 0.1)')
        else:
            ax4.plot([point[0], histories[0]['x'][i + 1][0]], [point[1], histories[0]['x'][i + 1][1]], c='r')

    for i, point in enumerate(histories[1]['x'][:-1]):
        # print(point)
        if i == 0:
            ax4.plot([point[0], histories[1]['x'][i + 1][0]], [point[1], histories[1]['x'][i + 1][1]], c='b', label='(-0.1, -0.1)')
        else:
            ax4.plot([point[0], histories[1]['x'][i + 1][0]], [point[1], histories[1]['x'][i + 1][1]], c='b')


    f_p1 = lambda x: np.sqrt(1-x**2)
    f_p2 = lambda x: -np.sqrt(1 - x ** 2)

    a = np.arange(-1, 1, 0.01)
    b_p1 = f_p1(a)
    b_p2 = f_p2(a)

    # print([(-1, 1)] + b_p1 + [(1,1)])
    pgon1 = plt.Polygon([(-1, 1)] + list(zip(a, b_p1)) + [(1,1)], closed=False, facecolor='w', alpha=0.7)
    pgon2 = plt.Polygon([(-1, -1)] + list(zip(a, b_p2)) + [(1,-1)], closed=False, facecolor='w', alpha=0.7)

    print(pgon1)
    ax4.add_patch(pgon1)
    ax4.add_patch(pgon2)

    ax4.contourf(X, Y, Z, 25, cmap=cm.rainbow)

    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()

    ax1.set_title("两种情况下x1迭代变化", fontproperties=font, fontsize=18)
    ax2.set_title("两种情况下x2迭代变化", fontproperties=font, fontsize=18)
    ax3.set_title("两种情况下函数值f迭代变化", fontproperties=font, fontsize=18)
    ax4.set_title("等值线下的两种情况的迭代情况", fontproperties=font, fontsize=18)

    plt.savefig(f"{filename}.svg")
    plt.show()


def plot_result_v2(history, f, filename="test"):

    X = np.arange(0, 1.1, 0.1)
    Y = np.arange(0, 1.1, 0.1)
    X, Y = np.meshgrid(X, Y)
    def fun(x, y):
        return f([x, y])
    Z = fun(X, Y)

    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    # ax3 = fig.add_subplot(223, projection='3d')
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    x01 = [x_[0] for x_ in history["x"]]
    x02 = [x_[1] for x_ in history["x"]]
    print('x', len(x01), x01)
    ax1.plot(range(len(x01)), x01, c='b', label='(0.1, 0.1)')
    ax2.plot(range(len(x02)), x02, c='b', label='(0.1, 0.1)')

    # x11 = [x_[0] for x_ in histories[1]["x"]]
    # x12 = [x_[1] for x_ in histories[1]["x"]]
    # ax1.plot(range(len(x11)), x11, c='b', label='(-0.1, -0.1)')
    # ax2.plot(range(len(x12)), x12, c='b', label='(-0.1, -0.1)')

    ax3.plot(range(len(history['f(x)'])), history['f(x)'], c='b', label='(0.1, 0.1)')
    # ax3.plot(range(len(histories[1]['f(x)'])), histories[1]['f(x)'], c='b', label='(-0.1, -0.1)')

    # ax3.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.rainbow)
    # ax3.view_init(elev=20,  # 仰角
    #               azim=10  # 方位角
    #               )

    for i, point in enumerate(history['x'][:-1]):
        # print(point)
        if i==0:
            ax4.plot([point[0], history['x'][i + 1][0]], [point[1], history['x'][i + 1][1]], c='b', label='(0.1, 0.1)')
        else:
            ax4.plot([point[0], history['x'][i + 1][0]], [point[1], history['x'][i + 1][1]], c='b')

    # for i, point in enumerate(histories[1]['x'][:-1]):
    #     # print(point)
    #     if i == 0:
    #         ax4.plot([point[0], histories[1]['x'][i + 1][0]], [point[1], histories[1]['x'][i + 1][1]], c='b', label='(-0.1, -0.1)')
    #     else:
    #         ax4.plot([point[0], histories[1]['x'][i + 1][0]], [point[1], histories[1]['x'][i + 1][1]], c='b')
    f_p1 = lambda x: np.sqrt(1-x**2)
    # f_p2 = lambda x: -np.sqrt(1 - x ** 2)

    a = np.arange(1, 0, -0.01)
    print(a)
    b_p1 = f_p1(a)
    # b_p2 = f_p2(a)

    # print([(-1, 1)] + b_p1 + [(1,1)])
    pgon1 = plt.Polygon([(0, 1), (0, 3), (3, 3), (3, 0), (1, 0)] + list(zip(a, b_p1)), closed=False, facecolor='w', alpha=0.7)
    # pgon2 = plt.Polygon([(-1, -1)] + list(zip(a, b_p2)) + [(1,-1)], closed=False, facecolor='w', alpha=0.7)

    print(pgon1)
    ax4.add_patch(pgon1)
    # ax4.add_patch(pgon2)

    ax4.contourf(X, Y, Z, 25, cmap=cm.rainbow)

    # ax1.legend()
    # ax2.legend()
    # ax3.legend()
    # ax4.legend()

    ax1.set_title("x1迭代变化", fontproperties=font, fontsize=18)
    ax2.set_title("x2迭代变化", fontproperties=font, fontsize=18)
    ax3.set_title("函数值f迭代变化", fontproperties=font, fontsize=18)
    ax4.set_title("等值线下的迭代情况", fontproperties=font, fontsize=18)

    plt.savefig(f"{filename}.svg")
    plt.show()


if __name__ == '__main__':
    variables = [x, y]
    n = 2

    ff = -1 * x * y

    d = [sp.var(f"d{i}") for i in range(n)]

    # l = sp.Matrix([sp.var(f"l{i}") for i in range(n)]).reshape((n, 1))
    g = sp.Matrix([[x ** 2 + y ** 2 - 1]])
    l = [sp.var(f"l{i}") for i in range(len(g))]
    # lagrange = ff + l.T.dot(g)[0][0]
    max_iterations = 100
    start_default = 0.1 * np.ones((len(variables), 1))
    start_default2 = np.zeros((len(g), 1))
    # max_iterations = 31
    history = {}
    f = lambda x : -1 * np.exp(-2*np.log(2)*((x[0]-0.008)/0.854))*(np.sin(5*np.pi*(x[0]**0.75-0.05)))**6
    from sko.SA import SimulatedAnnealingBase
    import numpy as np
    sa = SimulatedAnnealingBase(func=f, x0=[1], T_max=10, T_min=1e-9, L=10, max_stay_counter=200)
    best_x, best_y = sa.run()
    print(best_x, best_y)
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    # ax3 = fig.add_subplot(223, projection='3d')
    ax3 = fig.add_subplot(133)
    ax1.plot(range(len(sa.x_history)), sa.x_history, label='x')
    ax2.plot(range(len(sa.y_history)), sa.y_history, label='y')
    ax3.plot(range(len(sa.T_history)), sa.T_history, label='T')
    print(len(sa.x_history))
    ax1.set_xlabel("iterations", fontsize=16)
    ax2.set_xlabel("iterations", fontsize=16)
    ax3.set_xlabel("iterations", fontsize=16)

    ax1.set_ylabel("x", fontsize=16)
    ax2.set_ylabel("y", fontsize=16)
    ax3.set_ylabel("T", fontsize=16)

    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.tight_layout()
    plt.savefig('opt8-1.svg')
    plt.show()


    f = lambda x : (2186-(x[0]**2+x[1]-11)**2-(x[0]+x[1]**2-7)**2)/2186

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    X = np.arange(-6, 6.1, 0.1)
    Y = np.arange(-6, 6.1, 0.1)
    X, Y = np.meshgrid(X, Y)
    def fun(x, y):
        return f([x, y])
    Z = fun(X, Y)

    ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.rainbow)
    ax1.view_init(elev=10,  # 仰角
                  azim=10  # 方位角
                  )
    ax2.contourf(X, Y, Z, 80, cmap=cm.rainbow)
    CS = ax2.contour(X, Y, Z, 80)
    ax2.clabel(CS, inline=True, fontsize=12)

    plt.savefig('opt8-2.svg')
    plt.show()

    fun = lambda x:-1*f(x)
    from sko.GA import GA
    ga = GA(func=fun, n_dim=2, size_pop=100, max_iter=50, prob_mut=0.001,  lb=[-6, -6], ub=[6, 6], precision=1e-6)
    best_x, best_y = ga.run()
    print('best_x:', best_x, '\n', 'best_y:', best_y)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    # print(np.asarray(ga.x_history).shape)
    x = np.asarray(ga.x_history)
    y = -np.asarray(ga.y_history)
    x1_mean, x2_mean, x1_std, x2_std = x[:,:,0].mean(axis=1), x[:,:,1].mean(axis=1), x[:,:,0].std(axis=1), x[:,:,1].std(axis=1)
    y_mean, y_std = y.mean(axis=1), y.std(axis=1)
    r1_x1 = list(map(lambda x: x[0] - x[1], zip(x1_mean, x1_std)))
    r2_x1 = list(map(lambda x: x[0] + x[1], zip(x1_mean, x1_std)))

    r1_x2 = list(map(lambda x: x[0] - x[1], zip(x2_mean, x2_std)))
    r2_x2 = list(map(lambda x: x[0] + x[1], zip(x2_mean, x2_std)))

    r1_y = list(map(lambda x: x[0] - x[1], zip(y_mean, y_std)))
    r2_y = list(map(lambda x: x[0] + x[1], zip(y_mean, y_std)))

    ax1.plot(range(len(ga.x_history)), x1_mean, label='x1')
    ax1.plot(range(len(ga.x_history)), x2_mean, label='x2')

    ax2.plot(range(len(ga.y_history)), y_mean, label='y', color='r')

    ax1.fill_between(range(len(x)), r1_x1, r2_x1, alpha=0.2)
    ax1.fill_between(range(len(x)), r1_x2, r2_x2, alpha=0.2)
    ax2.fill_between(range(len(x)), r1_y, r2_y, alpha=0.2, color='r')

    print(len(ga.x_history))
    ax1.legend()
    ax2.legend()

    ax1.set_xlabel("iterations", fontsize=16)
    ax2.set_xlabel("iterations", fontsize=16)
    ax1.set_ylabel("x", fontsize=16)
    ax2.set_ylabel("y", fontsize=16)
    plt.savefig('opt8-3.svg')
    plt.show()