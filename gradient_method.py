# steepest_descent_method_v2
# steepest_descent_method_v2(with decay)
# fr_method
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib import cm
from matplotlib.font_manager import FontProperties
from sympy.abc import x, y

font = FontProperties(fname="SimHei.ttf")


def evalf(f, subs):
    return np.array(f.evalf(subs=subs)).astype(np.float64)


def get_subs(array, variables):
    return {str(var): value for var, value in zip(variables, array.flatten())}


def fr_method(f, variables, variables_0, A, max_iterations=100):
    history = defaultdict(list)
    current_values = variables_0

    gradient = sp.Matrix([sp.diff(f, var) for var in variables])

    beta = 0
    gradient_before = evalf(gradient, get_subs(current_values, variables))
    d_before = -1 * gradient_before
    for i in range(max_iterations):
        # 打印结果
        v1 = i
        v2 = current_values.flatten()
        v3 = evalf(f, get_subs(current_values, variables)).flatten()
        v4 = evalf(gradient, get_subs(current_values, variables)).flatten()
        v5 = np.linalg.norm(v4)
        str_1 = v1
        str_2 = "({:.3f},{:.3f})".format(*v2)
        str_3 = "{:.3f}".format(v3[0])
        str_4 = "({:.3f},{:.3f})".format(*v4)
        str_5 = "{:.3f}".format(v5)
        print(f"| ${str_1}$ | ${str_2}$ | ${str_3}$ | ${str_4}$ | ${str_5}$ |")

        history["x"].append(v2)
        history["y"].append(v3[0])

        subs = get_subs(current_values, variables)
        gradient_ = evalf(gradient, subs)

        if i == 0:
            beta = 0
        else:
            beta = (np.linalg.norm(gradient_) / np.linalg.norm(gradient_before)) ** 2

        d = -1 * gradient_ + beta * d_before
        lam = -1 * (gradient_.T.dot(d) / d.T.dot(A).dot(d))
        current_values = current_values + lam * d
        gradient_before = gradient_
        d_before = d
    return history


def steepest_descent_method(f, variables, variables_0, max_iterations=100):
    history = defaultdict(list)
    current_values = variables_0

    gradient = sp.Matrix([sp.diff(f, var) for var in variables])

    for i in range(max_iterations):
        # 打印结果
        v1 = i
        v2 = current_values.flatten()
        v3 = evalf(f, get_subs(current_values, variables)).flatten()
        v4 = evalf(gradient, get_subs(current_values, variables)).flatten()
        str_1 = v1
        str_2 = "({:.3f},{:.3f})".format(*v2)
        str_3 = "{:.3f}".format(v3[0])
        str_4 = "({:.3f},{:.3f})".format(*v4)
        print(f"| ${str_1}$ | ${str_2}$ | ${str_3}$ | ${str_4}$ |")

        history["x"].append(v2)
        history["y"].append(v3[0])

        subs = get_subs(current_values, variables)
        gradient_ = evalf(gradient, subs)

        current_values = current_values - gradient_ / np.linalg.norm(gradient_)
    return history


def steepest_descent_method_v3(f, variables, variables_0, max_iterations=100):
    history = defaultdict(list)
    current_values = variables_0

    gradient = sp.Matrix([sp.diff(f, var) for var in variables])

    for i in range(max_iterations):
        # 打印结果
        v1 = i
        v2 = current_values.flatten()
        v3 = evalf(f, get_subs(current_values, variables)).flatten()
        v4 = evalf(gradient, get_subs(current_values, variables)).flatten()
        str_1 = v1
        str_2 = "({:.3f},{:.3f})".format(*v2)
        str_3 = "{:.3f}".format(v3[0])
        str_4 = "({:.3f},{:.3f})".format(*v4)
        print(f"| ${str_1}$ | ${str_2}$ | ${str_3}$ | ${str_4}$ |")

        history["x"].append(v2)
        history["y"].append(v3[0])

        subs = get_subs(current_values, variables)
        gradient_ = evalf(gradient, subs)

        current_values = current_values - gradient_ / np.linalg.norm(gradient_)
    return history


def steepest_descent_method_v2(f, variables, variables_0, max_iterations=100, decay_rate=1):
    assert decay_rate >= 0
    history = defaultdict(list)
    current_values = variables_0

    gradient = sp.Matrix([sp.diff(f, var) for var in variables])
    d = np.zeros((len(variables), 1))

    for i in range(max_iterations):
        step_size = decay_rate ** (i / (max_iterations - 1))

        # 打印结果
        v1 = i
        v2 = current_values.flatten()
        v3 = evalf(f, get_subs(current_values, variables)).flatten()
        v4 = evalf(gradient, get_subs(current_values, variables)).flatten()
        v5 = format(np.linalg.norm(v4), '.3f')
        v6 = format(step_size, '.3f')
        str_1 = v1
        str_2 = "({:.3f},{:.3f})".format(*v2)
        str_3 = "{:.3f}".format(v3[0])
        str_4 = "({:.3f},{:.3f})".format(*v4)
        str_5 = v5
        str_6 = v6
        print(f"| ${str_1}$ | ${str_2}$ | ${str_3}$ | ${str_4}$ | ${str_5}$ | ${str_6}$ |")

        history["x"].append(v2)
        history["y"].append(v3[0])
        history["step_size"].append(v5)

        subs = get_subs(current_values, variables)
        gradient_ = evalf(gradient, subs)
        d = step_size * gradient_ / np.linalg.norm(gradient_)
        current_values = current_values - d
    return history


def plot_result(history, f, filename="test"):
    X = np.arange(0, 4, 0.1)
    Y = np.arange(0, 4, 0.1)
    X, Y = np.meshgrid(X, Y)
    Z = f(X, Y)

    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224)

    x1 = [x_[0] for x_ in history["x"]]
    x2 = [x_[1] for x_ in history["x"]]
    ax1.plot(range(len(x1)), x1)
    ax2.plot(range(len(x2)), x2)

    ax3.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.rainbow)
    ax3.view_init(elev=20,  # 仰角
                  azim=10  # 方位角
                  )

    for i, point in enumerate(history['x'][:-1]):
        ax4.plot([point[0], history['x'][i + 1][0]], [point[1], history['x'][i + 1][1]], c='k')

    ax4.contourf(X, Y, Z, 100, cmap=cm.rainbow)

    ax1.set_title("x1值迭代变化", fontproperties=font, fontsize=18)
    ax2.set_title("x2值迭代变化", fontproperties=font, fontsize=18)
    ax3.set_title("函数surf", fontproperties=font, fontsize=18)
    ax4.set_title("锯齿现象的呈现", fontproperties=font, fontsize=18)

    plt.savefig(f"{filename}.svg")
    plt.show()


if __name__ == '__main__':
    variables = [x, y]
    ff = (x - 2) ** 2 + 4 * (y - 3) ** 2
    f = lambda a, b: (a - 2) ** 2 + 4 * (b - 3) ** 2

    variables_0 = np.zeros((len(variables), 1))
    max_iterations = 50 + 1
    history = {}
    history["steepest_descent_method"] = steepest_descent_method_v2(ff, variables, variables_0,
                                                                    max_iterations=max_iterations)
    history["steepest_descent_method_v2"] = steepest_descent_method_v2(ff, variables, variables_0,
                                                                       max_iterations=max_iterations, decay_rate=0.1)
    # history["steepest_descent_method_v3"] = steepest_descent_method_v2(ff, variables, variables_0, max_iterations=max_iterations, gamma=0.1)
    history["fr_method"] = fr_method(ff, variables, variables_0, A=2 * np.array([[1, 0], [0, 4]]),
                                     max_iterations=max_iterations)

    plot_result(history["steepest_descent_method"], f, filename="steepest_descent_method_v1" + str(max_iterations))
    plot_result(history["steepest_descent_method_v2"], f, filename="steepest_descent_method_v2" + str(max_iterations))
    # plot_result(history["steepest_descent_method_v3"], f, filename="steepest_descent_method")
    plot_result(history["fr_method"], f, filename="fr" + str(max_iterations))
