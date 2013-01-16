"""
Module defining the TensorStorage class.
"""

__authors__ = "Ian Goodfellow"
__copyright__ = "(c) 2013, Universite de Montreal"
__license__ = "3-clause BSD License"
__contact__ = "theano-dev <theano-dev@googlegroups.com>"

import copy

from theano.compile.storage import Storage

class TensorStorage(Storage):
    """
    A Storage class that stores tensor data.
    Currently, this Storage object only provides access to its internal
    data as TensorType.
    It is planned to eventually allow read and write access using both
    TensorType and CudaNDArrayType data.
    """

    # Constants to pass as the "kind" argument of the get_value method
    CURRENT = 0
    NUMPY = 1
    CUDA = 2

    def __init__(self, tensor_type):
        """
        :param tensor_type: TensorType instance defining the type of
            data provided when the stored data is accessed as a TensorType.
        """

        self.tensor_type = tensor_type
        self.numpy = None


    def get_value(self, borrow=False, kind=TensorStorage.CURRENT):
        """
        Return the value stored in this object as a numpy ndarray.

        :param borrow: If True, may return a direct reference to an internal
            data structure (note that there may be one valid internal data
            structure at the same time).
        :param kind: integer code specifying which kind of data to return:
            TensorConstant.CURRENT: Return any kind of data that the Storage
                object is currently storing. If both numpy and cuda structures
                are simultaneously valid, defaults to numpy.
            TensorConstant.NUMPY: Return the data as a numpy ndarray
            TensorConstant.CUDA: Return the data as a cuda ndarry

        Note that you are only guaranteed to get a reference rather than a copy
        if you specify both borrow=True and kind=TensorStorage.CURRENT, but in
        this case the return type is not predictable.
        """

        if kind not in [TensorStorage.CURRENT, TensorStorage.NUMPY,
                TensorStorage.CUDA]:
            raise ValueError("kind must be TensorStorage.{CURRENT,NUMPY,CUDA}"
                    ", got " + str(kind))

        numpy = self.numpy

        if kind == TensorStorage.CUDA:
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

        raise NotImplementedError(str(type(self))+
                "does not implement set_value.")

        if not borrow:
            value = copy.deepcopy(value)

        if value is None:
            self.numpy = value
        else:
            if hasattr(self.tensor_type, 'filter_inplace') and self.numpy is not None:
                self.numpy = self.type.filter_inplace(value, self.numpy, ** self.kwargs)
            else:
                self.numpy = self.type.filter(value, ** self.filter_kwargs)

class TensorStorageVariable(Variable, _tensor_py_operators):
    """
    A Variable whose non-symbolic value is TensorStorage.
    This is mostly a temporary construct to ease the transition
    to a new interface, where TensorStorage can be used as a value
    of a TensorVariable directly.
    """

    def __init__(self, name, type, value, strict,
                 allow_downcast=None, container=None):
        """
        :param name: The name for this variable (see `Variable`).

        :param type: The type for this variable (see `Variable`).

        :param value: A value to associate with this variable (a new
        container will be created).

        :param strict: True -> assignments to .value will not be cast
        or copied, so they must have the correct type.

        :param allow_downcast: Only applies if `strict` is False.
        True -> allow assigned value to lose precision when cast during assignment.
        False -> never allow precision loss.
        None -> only allow downcasting of a Python float to a scalar floatX.

        :param container: The container to use for this
        variable. Illegal to pass this as well as a value.

        :note: For more user-friendly constructor, see `shared`

        """
        Variable.__init__(self, type=type, name=name,
                                             owner=None, index=None)
        self.storage = TensorStorage(type, strict, allow_downcast)
        self.storage.set_value(value)
