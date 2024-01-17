import math

eps = 0.0000000001

def solve_quadratic(a, b, c):
    # This solves A*x*x + B*x + C = 0 for x
    # TODO: REALLY STRESS TEST THIS BECAUSE IT'S SO IMPORTANT
    # TODO: Handle cases where each of A, B, C are basically 0
    d = b*b - 4*a*c
    if d < 0:
        r1 = None
        r2 = None
    elif a == 0:
        # This is really just a linear equation
        r1 = -c/b
        r2 = None
    else:
        if b > 0:
            u = -b - math.sqrt(d)
        else:
            u = -b + math.sqrt(d)
        r1 = u/(2*a)
        r2 = 2*c/u
    return r1, r2

def solve_quadratic_old(A, B, C):
    # This solves A*x*x + B*x + C = 0 for x
    # TODO: REALLY STRESS TEST THIS BECAUSE IT'S SO IMPORTANT
    # TODO: Handle cases where each of A, B, C are basically 0
    D = B*B - 4*A*C
    r1, r2 = None, None
    if D < 0:
        return r1, r2
    elif abs(A) < eps:
        r1 = -C/B # We're solving a linear function. There might be a second root that's suuuuuuper far from t=0 so we don't care about it
    elif abs(A) < tad:
        # A is probably smaller than B or C, so use alternative quadratic formula to get better numerical stability
        r1 = (2*C)/(-B + math.sqrt(D))
        r2 = (2*C)/(-B - math.sqrt(D))
    elif D > 0:
        r1 = (-B - math.sqrt(D))/(2*A)
        r2 = (-B + math.sqrt(D))/(2*A)
    elif D == 0:
        r1 = -B/(2*A)
    return r1, r2

a = 4.3162933554e-11
b = 1.0361118521e6
c = -4.5813932360e5

print(solve_quadratic(a, b, c))
print(solve_quadratic_old(a, b, c))