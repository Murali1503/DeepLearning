import sympy as sm

x = sm.symbols("x")
fx = 2 * x**2

print(sm.diff(fx))
