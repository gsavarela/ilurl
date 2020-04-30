"""Provides decorators for classes' properties"""
import functools


def lazy_property(function):
    """Evaluates the function once creating a static method

    It stores the result in a member named after the decorated
    function (prepended with a prefix) and returns this value
    on any subsequent calls.

    Params:
        * function: function
            method or property to be decorated
    Returns:
        * lazy_property: type<function>

    References:
        https://danijar.com/structuring-your-tensorflow-models/

    Decorators:
        functools.wraps
    """

    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


def delegate_property(function):
    """Delegates the function to another object that responds to it.

    It uses method lookup in order to find an object (delegate) that
    answers to that function. Once this delegate is found, it
    becomes the permanent object being invoked

    Params:
        * function: <type>

    Returns:
        * decorated_function: <type>

    Decorators:
        functools.wraps

    Examples:
        @delegate_property
        def foo(self)           <===>     def foo(self)
            pass                            returns self.delegate.foo()
    """

    # This attribute will point to the delegate
    attribute = '_delegate_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            # Find delegate and assign
            # Non default attributes
            lookup_attributes = \
                set(self.__dict__.keys()) - set(object.__dict__.keys())

            for attr_ in lookup_attributes:
                # Get attribute -- this is a delegate candidate
                delegate_ = getattr(self, attr_)
                # if delegate object is an implementor of the function
                if hasattr(delegate_, function.__name__):
                    # Creates the lazy property pointing to 
                    # the delate implamentation
                    setattr(self, attribute, getattr(delegate_, function.__name__))
                    break
        return getattr(self, attribute)
    return decorator
