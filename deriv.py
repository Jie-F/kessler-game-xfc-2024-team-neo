from sympy import symbols, cos, sin, Abs, diff

# Define the symbols
theta = symbols('theta', real=True)
avx, vb, t_0, ay, avy, theta_0, tr, ax = symbols('avx vb t_0 ay avy theta_0 tr ax', real=True)

# Define the function
f = (avx - vb * cos(theta)) * (vb * t_0 * sin(theta) - ay - avy * Abs(theta - theta_0) / tr) - \
    (avy - vb * sin(theta)) * (vb * t_0 * cos(theta) - ax - avx * Abs(theta - theta_0) / tr)

# Compute the first and second derivatives with respect to theta
#f_first_derivative = diff(f, theta)
f_second_derivative = diff(f, theta, 2)

# Calculate the ratio of the second derivative to the first derivative
#ratio = f_second_derivative / f_first_derivative
#simplified_ratio = ratio.simplify()
f_second_derivative.simplify()
#print(simplified_ratio)
print(f_second_derivative)
