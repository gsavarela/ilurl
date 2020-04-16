import pdb
from pathlib import Path
from time import time
import json
import sys
import multiprocessing as mp
from ilurl.utils.context_managers import PipeGuard


def benchmarked(fnc):
    """Times execution of fnc, storing on folder if path exists

        Parameters:
        ----------
        * fnc: function
            An anonymous function decorated by the user

        Returns:
        -------
        * fnc: function
            An anonymous function that will be timed
    """
    _data = {}
    def f(*args, **kwargs):
        _data['start'] = time()
        res = fnc(*args, **kwargs)
        _data['finish'] = time()
        _data['elapsed'] = _data['finish'] - _data['start']

        # if res is a valid sys path
        if Path(str(res)).exists():
            target_path = Path(str(res)) / 'time.json'
            with target_path.open('w') as f:
                json.dump(_data, f)
        else:
            print(f'''\tChronological:
                        ---------------
                      \t start:{_data['start']}
                      \t finish:{_data['finish']}
                      \t elapsed:{_data['elapsed']}\n\n''')
        return res
    return f


def processable(fnc):
    """Supresses stdout during fnc execution writing output only

        Parameters:
        ----------
        * fnc: function
            An anonymous function decorated by the user

        Returns:
        -------
        * fnc: function
            An anonymous function that will have stdout supressed
    """
    def f(*args, **kwargs):
        with PipeGuard():
            res = fnc(*args, **kwargs)
        # send result to the pipeline
        sys.stdout.write(res)
        return res
    return f


# TODO: not working -- lock has to be shared?
# implemented with a global lock on jobs/train.py
def delayable(lock):
    """delayable receives a lock and delays fnc execution 

        Parameters:
        ----------
        *   lock: multiprocessing.Lock
                An instance of multiprocessing lock

        Returns:
        -------
        * fnc: function
            An anonymous function decorated by the user
    """
    def delay(fnc):
        """delays execution by 1 sec.

            Parameters:
            -----------
            * fnc: function
                An anonymous function decorated by the user

            Returns:
            -------
            * fnc : function
                An anonymous function to be executed 1 sec. after
                calling
        """
        def f(*args, **kwargs):
            LOCK.acquire()
            try:
                time.sleep(1)
            finally:
                LOCK.release()
            return fnc(*args, **kwargs)
        return fnc
    return delay

