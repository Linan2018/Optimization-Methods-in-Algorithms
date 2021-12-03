import numpy as np
import matplotlib.pyplot as plt

# 黄金分割法
f = lambda x: min(x/2,2-(x-3)**2,2-x/2) * -1

fig = plt.figure()
xx = np.linspace(0,8,100)
plt.plot(xx, [f(x) for x in xx])
plt.xlabel('x', fontsize=16)
plt.ylabel('f(x)', fontsize=16)
plt.savefig('fx.svg')
plt.show()


l = 0
r = 8
rate = 0.618
for i in range(10):
    print('',i+1, l, r, f(l), f(r),'', sep='|')
    a = rate*l+(1-rate)*r
    b = (1-rate)*l+rate*r
    if f(a) > f (b):
        l = a
    else:
        r = b


print()

# 斐波那契法
f = lambda x: min(x/2,2-(x-3)**2,2-x/2) * -1
l = 0
r = 8
fib1 = 0
fib2 = 1
cur = fib1 + fib2
for i in range(10):
    rate = fib2 / cur
    print('', i+1, l, r, f(l), f(r),'', sep='|')
    a = rate*l+(1-rate)*r
    b = (1-rate)*l+rate*r
    if f(a) > f (b):
        l = a
    else:
        r = b
    fib1 = fib2
    fib2 = cur
    cur = fib1 + fib2


# 固定步长法
f_before = float('inf')
x_before = 0
for x in np.linspace(0, 8, num=5):
    if f(x) > f_before:
        break
    print(x, f(x))
    x_before = x
    f_before = f(x)

