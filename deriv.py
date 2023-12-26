from sympy import symbols, sin, cos, pi, Abs, diff
import sympy as sp

# Define the symbols
t_0, avx, avy, ax, ay, vb, theta, theta_0 = symbols('t_0 avx avy ax ay vb theta theta_0')

# Define the function
f = (t_0*avx - avx*(pi - Abs(theta - theta_0))/pi + ax)*(sin(theta) - avy/vb) - \
    (t_0*avy - avy*(pi - Abs(theta - theta_0))/pi + ay)*(cos(theta) - avx/vb)

# Calculate the derivative
derivative = diff(f, theta)

# Simplify the expression
simplified_derivative = sp.simplify(derivative)

print(simplified_derivative)

Piecewise(((avx*t_0 - avx + ax)*cos(theta_0) + (avy*t_0 - avy + ay)*sin(theta_0), Eq(theta, theta_0)), ((-(-avx*sin(theta) + avy*cos(theta))*((re(theta) - re(theta_0))*Derivative(re(theta), theta) + (im(theta) - im(theta_0))*Derivative(im(theta), theta)) + ((avx*(Abs(theta - theta_0) - pi) + pi*(avx*t_0 + ax))*cos(theta) + (avy*(Abs(theta - theta_0) - pi) + pi*(avy*t_0 + ay))*sin(theta))*Abs(theta - theta_0))/(pi*Abs(theta - theta_0)), True))
