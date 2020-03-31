"""Metaclass module to help enforce constraints on derived classes"""
__author__ = 'Guilherme Varela'
__date__ = '2020-03-25'

class MetaAgentQ(type):
    """AgentQ: type classes must implement those methods

    Methods:
    -------
    * act: tuple<int>
        encodes an agent's action

    * update: tuple<int>
        updates Q table values
    
    * stop: bool
        stops Q table updates

    * Q: dict<tuple,<tuple, float>>
        Q learning table with policy

    References:
        https://docs.python.org/3/reference/datamodel.html#metaclasses
        https://realpython.com/python-metaclasses/
    """
    def __new__(meta, name, base, body):
        agent_q_methods = ('act', 'update', 'stop', 'Q')
        for attr in agent_q_methods:
            if attr not in body:
                raise TypeError(f'AgentQ must implement {attr}')

        return super().__new__(meta, name, base, body)
