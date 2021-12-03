# newton_method lm_method bfgs_method dfp_method
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy.abc import x, y


def evalf(f, subs):
    return np.array(f.evalf(subs=subs)).astype(np.float64)


def get_subs(array, variables):
    return {str(var): value for var, value in zip(variables, array.flatten())}


# def newton_method_direction(current_values, variables):
#     subs = get_subs(current_values, variables)
#     gradient_ = evalf(gradient, subs)
#     h_ = evalf(h, subs)
#     # print(gradient_, h_, subs)
#     # print(h_)
#     d_k = -1 * np.linalg.inv(h_).dot(gradient_)

def newton_method(f, variables, variables_0, max_iterations=100):
    history = defaultdict(list)
    current_values = variables_0

    gradient = sp.Matrix([sp.diff(f, var) for var in variables])
    h = sp.hessian(f, variables)
    h_ = evalf(h, get_subs(current_values, variables))

    for i in range(max_iterations):
        # 打印结果
        v1 = i
        v2 = current_values.flatten()
        v3 = evalf(f, get_subs(current_values, variables)).flatten()
        v4 = evalf(gradient, get_subs(current_values, variables)).flatten()
        v5 = h_
        str_1 = v1
        str_2 = "({:.3f},{:.3f})".format(*v2)
        str_3 = "{:.3f}".format(v3[0])
        str_4 = "({:.3f},{:.3f})".format(*v4)
        str_5 = "\\left(\\begin{array}{ll}" + str(format(h_[0][0], '.3f')) + " & " + str(
            format(h_[0][1], '.3f')) + " \\\\ " + str(format(h_[1][0], '.3f')) + " & " + str(
            format(h_[1][1], '.3f')) + "\\end{array}\\right)"
        print(f"| ${str_1}$ | ${str_2}$ | ${str_3}$ | ${str_4}$ | ${str_5}$ |")

        history["x"].append(v2)
        history["y"].append(v3[0])

        subs = get_subs(current_values, variables)
        gradient_ = evalf(gradient, subs)
        h_ = evalf(h, subs)
        d = -1 * np.linalg.inv(h_).dot(gradient_)
        current_values = current_values + d
    return history


def lm_method(f, variables, variables_0, max_iterations=100, lambda_=1):
    history = defaultdict(list)

    current_values = variables_0

    gradient = sp.Matrix([sp.diff(f, var) for var in variables])
    h = sp.hessian(f, variables)

    for i in range(max_iterations):
        # 打印结果
        v1 = i
        v2 = current_values.flatten()
        v3 = evalf(f, get_subs(current_values, variables)).flatten()
        v4 = evalf(gradient, get_subs(current_values, variables)).flatten()
        v5 = evalf(h, get_subs(current_values, variables))
        str_1 = v1
        str_2 = "({:.3f},{:.3f})".format(*v2)
        str_3 = "{:.3f}".format(v3[0])
        str_4 = "({:.3f},{:.3f})".format(*v4)
        str_5 = "\\left(\\begin{array}{ll}" + str(format(v5[0][0], '.3f')) + " & " + str(
            format(v5[0][1], '.3f')) + " \\\\ " + str(format(v5[1][0], '.3f')) + " & " + str(
            format(v5[1][1], '.3f')) + "\\end{array}\\right)"
        print(f"| ${str_1}$ | ${str_2}$ | ${str_3}$ | ${str_4}$ | ${str_5}$ |")

        history["x"].append(v2)
        history["y"].append(v3[0])

        subs = get_subs(current_values, variables)
        gradient_ = evalf(gradient, subs)
        h_ = evalf(h, subs)
        d = -1 * (np.linalg.inv(lambda_ * np.identity(len(variables)) + h_)).dot(gradient_)
        current_values = current_values + d
    return history


def bfgs_method(f, variables, variables_0, max_iterations=100):
    history = defaultdict(list)

    current_values = variables_0

    h_hat = np.array([[110, -6], [-6, 18]])

    gradient = sp.Matrix([sp.diff(f, var) for var in variables])

    for i in range(max_iterations):
        # 打印结果
        v1 = i
        v2 = current_values.flatten()
        v3 = evalf(f, get_subs(current_values, variables)).flatten()
        v4 = evalf(gradient, get_subs(current_values, variables)).flatten()
        v5 = h_hat
        str_1 = v1
        str_2 = "({:.3f},{:.3f})".format(*v2)
        str_3 = "{:.3f}".format(v3[0])
        str_4 = "({:.3f},{:.3f})".format(*v4)
        str_5 = "\\left(\\begin{array}{ll}" + str(format(v5[0][0], '.3f')) + " & " + str(
            format(v5[0][1], '.3f')) + " \\\\ " + str(format(v5[1][0], '.3f')) + " & " + str(
            format(v5[1][1], '.3f')) + "\\end{array}\\right)"
        print(f"| ${str_1}$ | ${str_2}$ | ${str_3}$ | ${str_4}$ | ${str_5}$ |")

        history["x"].append(v2)
        history["y"].append(v3[0])

        subs = get_subs(current_values, variables)
        gradient_before = evalf(gradient, subs)

        d = -1 * np.linalg.inv(h_hat).dot(gradient_before)
        alpha = 1
        a_before = 100
        target_before = float('inf')
        flag = 0
        for a in np.linspace(0.01, 1, num=100):
            target = evalf(f, get_subs(current_values + a * d, variables))
            if target > target_before and flag:
                alpha = a_before
                flag = 0
                break
            target_before = target
            a_before = a
        alpha = 1
        s = alpha * d
        # print(s, np.linalg.norm(s))

        current_values = (current_values + s).copy()

        subs = get_subs(current_values, variables)
        gradient_ = evalf(gradient, subs).copy()
        q = (gradient_ - gradient_before).copy()
        h_hat = h_hat + q.dot(q.T) / (q.T.dot(s)) - h_hat.dot(s).dot(s.T).dot(h_hat.T) / (s.T.dot(h_hat).dot(s))

    return history


def dfp_method(f, variables, variables_0, max_iterations=100):
    history = defaultdict(list)
    current_values = variables_0

    d_hat = np.linalg.inv(np.array([[110, -6], [-6, 18]]))

    gradient = sp.Matrix([sp.diff(f, var) for var in variables])

    for i in range(max_iterations):
        # 打印结果
        v1 = i
        v2 = current_values.flatten()
        v3 = evalf(f, get_subs(current_values, variables)).flatten()
        v4 = evalf(gradient, get_subs(current_values, variables)).flatten()
        v5 = d_hat
        str_1 = v1
        str_2 = "({:.3f},{:.3f})".format(*v2)
        str_3 = "{:.3f}".format(v3[0])
        str_4 = "({:.3f},{:.3f})".format(*v4)
        str_5 = "\\left(\\begin{array}{ll}" + str(format(v5[0][0], '.3f')) + " & " + str(
            format(v5[0][1], '.3f')) + " \\\\ " + str(format(v5[1][0], '.3f')) + " & " + str(
            format(v5[1][1], '.3f')) + "\\end{array}\\right)"
        print(f"| ${str_1}$ | ${str_2}$ | ${str_3}$ | ${str_4}$ | ${str_5}$ |")

        history["x"].append(v2)
        history["y"].append(v3[0])

        subs = get_subs(current_values, variables)
        gradient_before = evalf(gradient, subs)

        d = -1 * d_hat.dot(gradient_before)

        # find alpha
        # c1 = 1e-4
        # c2 = 0.1
        alpha = 1
        a_before = 1
        target_before = float('inf')
        for a in np.linspace(0.01, 1, num=100):
            target = evalf(f, get_subs(current_values + a * d, variables))
            if target > target_before:
                alpha = a_before
                break
            target_before = target
            a_before = a
        alpha = 1
        s = alpha * d

        current_values = (current_values + s).copy()

        subs = get_subs(current_values, variables)
        gradient_ = evalf(gradient, subs)
        q = gradient_ - gradient_before
        d_hat = d_hat + s.dot(s.T) / (q.T.dot(s)) - d_hat.dot(q).dot(q.T).dot(d_hat) / (q.T.dot(d_hat).dot(q))

    return history


if __name__ == '__main__':
    variables = [x, y]
    ff = (x - 3) ** 4 + (x - 3 * y) ** 2
    variables_0 = np.zeros((len(variables), 1))
    max_iterations = 31
    history = {}
    history["newton_method"] = newton_method(ff, variables, variables_0, max_iterations=max_iterations)
    history["lm_method"] = lm_method(ff, variables, variables_0, max_iterations=max_iterations)
    history["bfgs_method"] = bfgs_method(ff, variables, variables_0, max_iterations=max_iterations)
    history["dfp_method"] = dfp_method(ff, variables, variables_0, max_iterations=max_iterations)

    # fig = plt.figure(figsize=(6, 6))
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    ax1, ax2 = axs[0], axs[1]
    for method, d in history.items():
        ax1.plot(np.log10(range(max_iterations)), np.log10(d["y"]), label=method)
        ax1.set_xlabel("log(iterations)", fontsize=16)
        ax1.set_ylabel("log(f(x))", fontsize=16)
        ax2.plot(range(max_iterations), d["y"], label=method)
        ax2.set_xlabel("iterations", fontsize=16)
        ax2.set_ylabel("f(x)", fontsize=16)
    ax1.legend()
    ax2.legend()
    plt.savefig('y对比.svg')
    plt.show()

    fig = plt.figure()
    marks = ['o', '^', 's', 'D']
    corlors = ['b', 'r', 'g', 'y']
    for i, (method, d) in enumerate(history.items()):
        x = np.concatenate(d['x'], axis=0).reshape((-1, 2))
        x1 = x[:, 0].flatten()
        x2 = x[:, 1].flatten()
        plt.scatter(range(max_iterations), x1, alpha=0.6, label=method + ' x1', marker=marks[i],
                    c=[corlors[i]] * max_iterations)
        plt.scatter(range(max_iterations), x2, alpha=0.6, label=method + ' x2', marker=marks[i],
                    c=[corlors[i]] * max_iterations)
    plt.legend()
    plt.xlabel('iterations')
    plt.savefig('x对比.svg')
    plt.show()
