from multiprocessing import Process, Queue
import json
import glob
import inspect
import pandas as pd

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
        fname = fname+'__'+func
        if postfix:
            fname += '_' + postfix.lower().replace(' ','-')
        return fname+'.json'

    def add(self, result):
        self.results.append(result)
        with open(self.fname, 'w') as f:
            json.dump(self.results, f, indent=2)

def load_results(fname):
    data = pd.DataFrame()
    for fn in glob.glob(fname):
        with open(fn, 'r') as f:
            result = json.load(f)
            data = data.append(result)
    return data
