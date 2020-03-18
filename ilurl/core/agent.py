


class Agent(object):

    def __init__(self):

        # Whether agent is training or not.
        self.stop = False

        # Store number of updates.
        self.updates_counter = 0

        # Learning rate.
        #self.apha

    @property
    def stop(self):
        """Stops exploring"""
        return self._stop

    @stop.setter
    def stop(self, stop):
        self._stop = stop    

    def get_action(self, state):
        raise NotImplementedError

    def update(self,
               state,
               action,
               reward,
               next_state):
        raise NotImplementedError


attributes / getters:
    - USED STATE + REWARD

    - explored

    - discrete space

    - discrete action

    - params

    - learnable parameters (Q -table or model weights)

methods:
    - get action