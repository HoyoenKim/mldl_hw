import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

x1 = np.linspace(-3, 1, 400)
x2 = np.linspace(1, 3, 400)

y1 = 2 + x1 - 2 * (x1 - 1)**2
y2 = 2 + x2

plt.plot(x1, y1, label='2 + x - 2 * (x - 1)^2', color='b')
plt.plot(x2, y2, label='2 + x', color='r')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.grid(True)
plt.title('curve')
plt.savefig('curve.png')
