from multiprocessing import Process, Queue
import json
import inspect

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
    ret = None if p.exitcode else q.get()
    return (ret, p.exitcode)

class SaveResults(object):
    def __init__(self, fname=None, postfix=''):
        if fname is None:
            self.fname = self._get_fname(postfix)
        else:
            self.fname = fname
        self.results = []

    def _get_fname(self, postfix):
        stack = inspect.stack()
        print(stack)
        assert len(stack) > 2, stack
        assert len(stack[-2]) > 3, stack[1]
        fname, func = stack[-2][1], stack[-2][3]
        fname = fname.replace('/', '__')+'__'+func
        if postfix:
            fname += '_' + postfix.lower().replace(' ','-')
        return fname+'.json'

    def add(self, result):
        self.results.append(result)
        with open(self.fname, 'w') as f:
            json.dump(self.results, f)
