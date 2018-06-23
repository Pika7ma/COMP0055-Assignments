import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
import time


def main(*args, **kwargs):
  dim = kwargs['dim']
  pool_sz = kwargs['pool_sz']
  net = tf.get_variable('input', [1, dim, dim, 1], initializer=tf.random_normal_initializer())
  net = tf.layers.max_pooling2d(net, pool_sz, 1, 'valid')
  config = tf.ConfigProto(
    device_count={'GPU': kwargs['GPU_num']}
  )

  with tf.Session(config=config) as sess:
    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()
    sess.run(tf.global_variables_initializer())
    res = sess.run(net)
    res = np.squeeze(res)
    # tl = timeline.Timeline(run_metadata.step_stats)
    # ctf = tl.generate_chrome_trace_format()
    # with open('timeline.json', 'w') as f:
    #     f.write(ctf)

  print(res)


if __name__ == '__main__':
  start = time.time()
  main(dim=1024, pool_sz=128, GPU_num=1)
  end = time.time()
  print(end - start)
