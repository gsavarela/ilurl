import os
import unittest

from ilurl.utils.serialize import Serializer


class TestSerialization(unittest.TestCase):
    class Foo(Serializer):
        def __init__(self, foo):
            self.foo = foo

    def setUp(self):
        '''Define the simplest test class'''
        script_path = os.path.dirname(os.path.realpath(__file__))
        tests_path = '/'.join(script_path.split('/')[:-1])
        self.dump_path = '{}/data/'.format(tests_path)

    def test_dump(self):
        f = self.Foo('Nerf')
        f.dump(self.dump_path, 'foo')

    def test_load(self):
        f = self.Foo('Nerf')
        f.dump(self.dump_path, 'foo')
        g = self.Foo.load(self.dump_path + 'foo.pickle')
        self.assertEqual(f.foo, g.foo)

    def tearDown(self):
        '''Remove pickles'''
        os.remove('{}{}'.format(self.dump_path, 'foo.pickle'))
