# Inspired from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/platform/benchmark.py
import inspect
import json

def report_benchmark(**kwargs):
      stack = inspect.stack()
      assert len(stack) > 1, stack
      assert len(stack[1]) > 3, stack[1]
      fname, func = stack[1][1], stack[1][3]
      kwargs += {'filename':fname, 'function':func}

      report_fname = fname.replace('/', '-')+':'+func+'.json'
      json.dump(kwargs)
      print('Results are written into '+report_fname)

      print(report_fname)
      print(json.dumpgs))
