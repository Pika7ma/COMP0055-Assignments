import numpy as np
import time


def main(*args, **kwargs):
  dim = kwargs['dim']
  pool_sz = kwargs['pool_sz']
  mu, sigma = 0, 0.1  # mean and standard deviation
  rand = np.random.RandomState(kwargs['seed'])
  mat = rand.normal(mu, sigma, [dim, dim])
  mat_ = np.zeros([dim - pool_sz + 1, dim - pool_sz + 1])
  for i in range(dim - pool_sz + 1):
    for j in range(dim - pool_sz + 1):
      mat_[i][j] = np.max(np.ravel(mat[i:i+pool_sz, j:j+pool_sz]))
  print(mat_)


if __name__ == '__main__':
  start = time.time()
  main(seed=0, dim=4096, pool_sz=5)
  end = time.time()
  print(end - start)