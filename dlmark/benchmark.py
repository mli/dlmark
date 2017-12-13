# Inspired from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/platform/benchmark.py

class Benchmark(object):
      """Abstract class that provides helper functions for running benchmarks.

      """

      def report_benchmark(self, iters=None, cpu_time=None, wall_time=None,
                           throughput=None, extras=None, name=None):
         print(cpu_time)



import unittest

class ResNet(unittest.TestCase):
    def test_hehe(self):
        print(1)

    def test_haha(self):
        print(2)
