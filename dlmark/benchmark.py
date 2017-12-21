from multiprocessing import Process, Queue
# import inspect
# import json

# def report_benchmark(**kwargs):
#      stack = inspect.stack()
#      assert len(stack) > 1, stack
#      assert len(stack[1]) > 3, stack[1]
#      fname, func = stack[1][1], stack[1][3]
#      kwargs += {'filename':fname, 'function':func}

#      report_fname = fname.replace('/', '-')+':'+func+'.json'
#      json.dump(kwargs)
#      print('Results are written into '+report_fname)

#      print(report_fname)
#      # print(json.dumpgs))


def run_with_separate_process(func, *args):
    def _process(queue, func, *args):
          res = func(*args)
          queue.put(res)
    q = Queue()
    p = Process(target=_process, args=(q, func, *args))
    p.start()
    p.join()
    return q.get()
