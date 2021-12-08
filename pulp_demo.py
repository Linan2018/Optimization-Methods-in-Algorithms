from pulp import *

prob = LpProblem("Problem", LpMaximize)
x1 = LpVariable("x1", lowBound=None, upBound=None)
x2 = LpVariable("x2", lowBound=None, upBound=None)

prob += 4 * x1 + 9 * x2 + 6
prob += (5 * x1 + 8 * x2 + 6 <= 12)

prob.solve()

print(x1.varValue)
print(x2.varValue)
print(value(prob.objective))


