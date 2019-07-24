'''
                    @author Guilherme Varela
                    @date 2019-07-19
                    This module helps storing the contents of a class
                    in pickle form. Furthermore, it replaces previous
                    json dump storage.

'''
import re

# dill allows to pickle lambda functions
import dill


class Serializer(object):
    '''This class performs loading and dumping'''

    @classmethod
    def load(cls, file_path):
        '''Recovers a new serialized object from disk
        '''
        ext = file_path.split('.')[-1]
        if ext != 'pickle':
            file_path += 'pickle'

        with open(file_path, 'rb') as f:
            serialized_instance = dill.load(f)

        return serialized_instance

    def dump(self, file_dir, filename=None):
        '''Serializes thru pickle'''

        if filename is None:
            filename = convert(self.__class__.__name__)

        if filename.split('.') != 'pickle':
            filename += '.pickle'

        if file_dir[-1] != '/':
            file_dir += '/'

        file_path = '{:}{:}'.format(file_dir, filename)
        with open(file_path, 'wb') as f:
            dill.dump(self, f)


def convert(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


# class Foo(Serializer):
#     def __init__(self, bar):
#         self.bar = bar
#
#     @classmethod
#     def load(cls, file_path):
#         super(Foo, cls).load(file_path)
#
#     def dump(self, file_dir, filename=None):
#         super(Foo, self).dump(file_dir, filename=filename)
#

if __name__ == '__main__':
    import os
    bar = Serializer()
    bar.foo = 'Guilherme'
    print('This is Bar#foo {}'.format(bar.foo))
    # f1 = Foo('Ahoy!')
    # print('This is Fooe#bar {}'.format(f1.bar))
    # f1.dump(os.getcwd())

    # b1 = Foo.load(os.getcwd() + '/foo.pickle')

    # print('This is Bar#bar {}'.format(b1.bar))
