import matplotlib.pyplot as plt
import numpy as np

# Define the wrapping function
def wrap_theta(theta, theta_0):
    if not (theta_0 - np.pi <= theta <= theta_0 + np.pi):
        theta = (theta - theta_0 + np.pi) % (2 * np.pi) - np.pi + theta_0
    return theta

# Test the function
theta_0 = 1 # You can change this to any value you like
theta_values = np.linspace(theta_0 - 4 * np.pi, theta_0 + 4 * np.pi, 1000)
wrapped_values = [wrap_theta(theta, theta_0) for theta in theta_values]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(theta_values, wrapped_values, label='Wrapped Values')
plt.axhline(y=theta_0 + np.pi, color='r', linestyle='--', label=r'$\theta_0 + \pi$')
plt.axhline(y=theta_0 - np.pi, color='g', linestyle='--', label=r'$\theta_0 - \pi$')
plt.xlabel('Original Theta')
plt.ylabel('Wrapped Theta')
plt.title('Wrapping Function Behavior')
plt.legend()
plt.grid(True)
plt.show()
