import time
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
            lock.acquire()
            try:
                time.sleep(1)
            finally:
                lock.release()
            return fnc(*args, **kwargs)
        return fnc
    return delay

