import sys
from ilurl.utils.context_managers import PipeGuard

def processable(fnc):
    def f(*args, **kwargs):
        with PipeGuard():
            res = fnc(*args, **kwargs)
        # send result to the pipeline
        sys.stdout.write(res)
        return res
    return f

