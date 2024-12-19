import numpy as np
from scipy.fft import dct

def barycentric_weights(x):
  """Computes the barycentric weights for interpolation on the grid given by x.
  The elements of x do not need to be sorted, but they must be unique."""

  N = len(x)
  w = np.ones(N)

  for j in range(1, N):
    for k in range(j):
      w[k] *= x[k] - x[j]
      w[j] *= x[j] - x[k]
  
  wmax = np.max(np.abs(w)) # scaling of the weights is meaningless might as well normalize
  w = wmax / w

  return w

def interpolation_matrix(x, t, w=None):
  """Computes the interpolation P. The columns of P are the Lagrange
  interpolating polynomials associated with collocation grid x evaluated at t.

  x: interpolation grid

  t: evaluation grid
  
  w: the barycentric weights associated with x. If w is not provided, w is
  computed using the function `barycentric_weights`.
  """

  if not w:
    w = barycentric_weights(x)
  
  N = len(x)
  M = len(t)

  P = np.zeros((M, N))
  for k in range(M):
    match = False
    for j in range(N):
      if np.allclose(t[k], x[j]):
        match = True
        P[k,j] = 1
    
    if not match:
      s = 0
      for j in range(N):
        z = w[j] / (t[k] - x[j])
        P[k,j] = z
        s += z
      P[k, :] /= s

  return P

def chebyshev_grid(a, b, N):
  """Computes the Chebyshev grid with N points on the interval [a, b]."""
  x = np.zeros(N)
  for i in range(N):
    x[i] = -np.cos( (2*i+1)*np.pi / (2*N) )
  
  x = (x + 1) * (b - a)/2 + a
  return x

def chebyshev_lobatto_grid(a, b, N):
  """Computes the Chebyshev-Lobatto grid with N points on the interval [a, b]."""
  x = np.zeros(N)
  for i in range(N):
    x[i] = -np.cos( i * np.pi / (N-1) )
  
  x = (x + 1) * (b - a)/2 + a
  return x

def diff_matrix(x, w=None):
  """Constructs the differentiation matrix for the collocation grid x."""

  if not w:
    w = barycentric_weights(x)

  N = len(x)
  D = np.zeros((N,N))
  for i in range(N):
    for j in range(N):
      if i != j:
        D[i,j] = w[j] / w[i] / (x[i] - x[j])
        D[i,i] -= D[i,j]
  return D
