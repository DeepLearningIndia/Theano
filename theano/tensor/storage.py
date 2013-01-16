"""
Module defining the TensorStorage class.
"""

__authors__ = "Ian Goodfellow"
__copyright__ = "(c) 2013, Universite de Montreal"
__license__ = "3-clause BSD License"
__contact__ = "theano-dev <theano-dev@googlegroups.com>"

import copy

from theano.compile.storage import Storage
from theano.gof.graph import Variable
from basic import _tensor_py_operators

# Constants to pass as the "kind" argument of the get_value method
CURRENT = 0
NUMPY = 1
CUDA = 2

class TensorStorage(Storage):
    """
    A Storage class that stores tensor data.
    Currently, this Storage object only provides access to its internal
    data as TensorType.
    It is planned to eventually allow read and write access using both
    TensorType and CudaNDArrayType data.
    """


    def __init__(self, tensor_type, strict, allow_downcast=None):
        """
        :param tensor_type: TensorType instance defining the type of
            data provided when the stored data is accessed as a TensorType.
        """

        self.tensor_type = tensor_type
        self.numpy = None

        self.filter_kwargs = {'strict' : strict,
                'allow_downcast' : allow_downcast }


    def get_value(self, borrow=False, kind=CURRENT):
        """
        Return the value stored in this object as a numpy ndarray.

        :param borrow: If True, may return a direct reference to an internal
            data structure (note that there may be one valid internal data
            structure at the same time).
        :param kind: integer code specifying which kind of data to return:
            CURRENT: Return any kind of data that the Storage
                object is currently storing. If both numpy and cuda structures
                are simultaneously valid, defaults to numpy.
            NUMPY: Return the data as a numpy ndarray
            CUDA: Return the data as a cuda ndarry

        Note that you are only guaranteed to get a reference rather than a copy
        if you specify both borrow=True and kind=TensorStorage.CURRENT, but in
        this case the return type is not predictable.
        """

        if kind not in [CURRENT, NUMPY, CUDA]:
            raise ValueError("kind must be TensorStorage.{CURRENT,NUMPY,CUDA}"
                    ", got " + str(kind))

        numpy = self.numpy

        if kind == CUDA:
            raise NotImplementedError("TensorStorage does not"
                    " support CUDA types yet.")

        if numpy is None:
            raise ValueError("Called get_value on empty storage.")

        if borrow:
            return numpy
        else:
            return copy.deepcopy(numpy)


    def set_value(self, value, borrow=False):
        """
        Set the value stored in this Storage object.

        :param borrow:
            True to use the value directly, potentially creating problems
            related to aliased memory.

        """

        if not borrow:
            value = copy.deepcopy(value)

        if value is None:
            self.numpy = value
        else:
            if hasattr(self.tensor_type, 'filter_inplace') and self.numpy is not None:
                self.numpy = self.tensor_type.filter_inplace(value, self.numpy, ** self.kwargs)
            else:
                self.numpy = self.tensor_type.filter(value, ** self.filter_kwargs)

class TensorStorageVariable(Variable, _tensor_py_operators):
    """
    A Variable whose non-symbolic value is TensorStorage.
    This is mostly a temporary construct to ease the transition
    to a new interface, where TensorStorage can be used as a value
    of a TensorVariable directly.
    """

    def __init__(self, name, type, value, strict,
                 allow_downcast=None, storage=None):
        """
        :param name: The name for this variable (see `Variable`).

        :param type: The type for this variable (see `Variable`).

        :param value: A TensorType value to associate with this variable
            (a new TensorStorage will be created).

        :param strict: True -> assignments to .value will not be cast
        or copied, so they must have the correct type.

        :param allow_downcast: Only applies if `strict` is False.
        True -> allow assigned value to lose precision when cast during assignment.
        False -> never allow precision loss.
        None -> only allow downcasting of a Python float to a scalar floatX.

        :param storage: The TensorStorage to use for this
                    variable. Illegal to pass this as well as a value.

        :note: For more user-friendly constructor, see `shared`

        """
        Variable.__init__(self, type=type, name=name,
                                             owner=None, index=None,
                                             storage=None)
        if storage is None:
            self.storage = TensorStorage(type, strict, allow_downcast)
            self.storage.set_value(value)
        else:
            assert value is None
            self.storage = storage

    def clone(self):
        """
        This is my (IG's) attempt at copying the clone method of shared variables.
        I'm not sure I got it right, because the original is not documented.
        Things I am unsure of:
            -what is the tag for a shared variable?
            -do we do deep or shallow copy of the storage? here I do shallow copy
            of the storage, because the shared variable does a shallow copy of the
            "container," but I don't understand the docstring for the "container"
            field at all.
        """
        cp = self.__class__(
                name=self.name,
                type=self.type,
                value=None,
                strict=None,
                storage=self.storage)
        cp.tag = copy.copy(self.tag)
        return cp

