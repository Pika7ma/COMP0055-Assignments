import torch
import time


def main(*args, **kwargs):

  # n, c, x, y
  input_tensor = torch.rand((1, 1, kwargs['dim'], kwargs['dim']))
  if kwargs['cuda']:
    input_tensor = input_tensor.cuda()
  max_pool = torch.nn.MaxPool2d(kwargs['pool_sz'], stride=1, padding=0)

  torch.cuda.synchronize()

  output = max_pool(input_tensor)
  print(output)


if __name__ == '__main__':
  start = time.time()
  main(dim=1024, pool_sz=128, cuda=True)
  end = time.time()
  print(end - start)
