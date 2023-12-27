import sympy as sp

# Define the symbols
theta = sp.symbols('theta')
avx, vb, t_0, ay, avy, theta_0, tr, ax = sp.symbols('avx vb t_0 ay avy theta_0 tr ax')

# Define the function
f = (avx - vb * sp.cos(theta)) * (vb * t_0 * sp.sin(theta) - ay - avy * sp.Abs(theta - theta_0) / tr) - \
    (avy - vb * sp.sin(theta)) * (vb * t_0 * sp.cos(theta) - ax - avx * sp.Abs(theta - theta_0) / tr)

# Compute the derivative with respect to theta
f_derivative = sp.diff(f, theta)
f_derivative.simplify()
print(f_derivative)
