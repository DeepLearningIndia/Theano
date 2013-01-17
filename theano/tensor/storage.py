"""
Module TensorStorage functionality.
"""

__authors__ = "Ian Goodfellow"
__copyright__ = "(c) 2013, Universite de Montreal"
__license__ = "3-clause BSD License"
__contact__ = "theano-dev <theano-dev@googlegroups.com>"

import copy
import warnings

from theano.compile.storage import Storage
from theano.gof.graph import Variable
from theano.gof import Apply
from theano.gof import Op
from theano.gof.type import Type

from basic import _tensor_py_operators
from basic import hashtype
from basic import TensorType

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
        self._numpy = None

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

        numpy = self._numpy

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

        Currently, this only works with numpy ndarrays.
        When cuda ndarrays are supported, then calling set_value with a numpy
        ndarray when there is a stored cuda ndarray will invalidate the stored
        cuda ndarray. A subsequent get_value call requesting a cuda ndarray will
        regenerate it by using a host-to-gpu transfer, at which point, there will
        be a valid version of both the numpy and the cuda ndarray in the storage
        object. The reverse is also true, ie, calling set_value with a cuda ndarray
        will invalidate any stored numpy ndarray.

        """

        if not borrow:
            value = copy.deepcopy(value)

        if value is None:
            self._numpy = value
        else:
            if hasattr(self.tensor_type, 'filter_inplace') and self._numpy is not None:
                self._numpy = self.tensor_type.filter_inplace(value, self._numpy, ** self.kwargs)
            else:
                self._numpy = self.tensor_type.filter(value, ** self.filter_kwargs)

class GetTensorFromStorage(Op):
    """
    An Op that accesses the tensor stored by a TensorStorage object
    in numpy ndarray format.
    """

    def make_node(self, x):
        if not isinstance(x, Variable):
            raise TypeError("Expected theano Variable.")

        if not isinstance(x.type, TensorStorageType):
            raise TypeError("Expected a Variable of type TensorStorageType.")

        output = x.storage.tensor_type.make_variable()

        return Apply(self, [x], [output])

    def grad(self, inputs, ograds):
        assert len(inputs) == 1
        assert len(ograds) == 1
        return ograds

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hashtype(self)

    def perform(self, node, inp, out):
        inp, = inp
        out, = out
        out[0] = inp.get_value(borrow=True, kind=NUMPY)

get_tensor_from_storage = GetTensorFromStorage()

class TensorStorageType(Type):
    """
    A Type representing values that are TensorStorage.
    This is mostly a temporary construct to ease the transition
    to a new interface, where TensorStorage can be used as a value
    of a TensorVariable directly.
    """
    def __init__(self, tensor_type):
        """
        :param tensor_type: A TensorType instance defining the type
            of tensors that this Type's values are allowed to store.
        """

        if not isinstance(tensor_type, TensorType):
            raise TypeError("Expected TensorType, got " +
                    str(type(tensor_type)))

        self.tensor_type = tensor_type


    def filter(self, data, strict=False, allow_downcast=None):
        """Convert `data` to something which can be associated to a
        `TensorVariable`.

        This function is not meant to be called in user code.  It is for
        `Linker` instances to use when running a compiled graph.
        """
        # Explicit error message when one accidentally uses a Variable as
        # input (typical mistake, especially with shared variables).
        if isinstance(data, Variable):
            raise TypeError(
                    'Expected a TensorStorage, but found a Variable: '
                    'maybe you are trying to call a function on a (possibly '
                    'shared) variable instead of a numeric array?')

        if not isinstance(data, TensorStorage):
            raise TypeError("Expected TensorStorage, got " +
                    str(type(data)))

        if not isinstance(data.tensor_type, self.tensor_type):
            raise TypeError("Expected TensorStorage storing " +
                    str(self.tensor_type) + ", got TensorStorage storing " +
                    str(data.tensor_type))

        return data

    def filter_variable(self, other):
        """
        Convert a symbolic Variable into this type, if possible.
        (No conversions are currently possible)
        """
        if other.type == self:
            return other

        raise TypeError(
                'Cannot convert Type %(othertype)s '
                '(of Variable %(other)s) into Type %(self)s. '
                'You can try to manually convert %(other)s into a %(self)s.'
                % dict(
                    othertype=other.type,
                    other=other,
                    self=self)
                )

    def value_validity_msg(self, a):
        try:
            self.filter(a, strict=True)
        except Exception, e:
            return str(e)
        return "value is valid"

    def __eq__(self, other):
        """Compare True iff other is the same kind of TensorStorage"""
        return type(self) == type(other) and \
            self.tensor_type == other.tensor_type

    @staticmethod
    def values_eq(a, b, force_same_dtype=True):
        return a.tensor_type.values_eq(a.get_value(), b.get_value(),
                force_same_dtype)

    @staticmethod
    def values_eq_approx(a, b, allow_remove_inf=False, allow_remove_nan=False):
        """
        :param allow_remove_inf: If True, when there is an inf in a.get_value(),
                                 we allow any value in b.get_value() in that
                                 position.
                                 Even -inf
        :param allow_remove_nan: If True, when there is a nan in a.get_value(),
                                 we allow any value in b.get_value() in that position.
                                 Even +-inf
        """
        raise NotImplementedError("The below is a copy-paste of TensorType.values_eq_approx")
        if isinstance(a, numpy.ndarray) and isinstance(b, numpy.ndarray):
            if a.shape != b.shape:
                return False
            if a.dtype != b.dtype:
                return False
            if 'int' in str(a.dtype):
                return numpy.all(a == b)
            else:
                # work around a numpy.allclose bug:
                # http://projects.scipy.org/numpy/ticket/1672
                if a.ndim == 0 and numpy.isinf(a):
                    a = a.reshape(1)
                    b = b.reshape(1)

                cmp = _allclose(a, b)
                if cmp:
                    # Numpy claims they are close, this is good enough for us.
                    return True
                # Numpy is unhappy, but it does not necessarily mean that a and
                # b are different. Indeed, Numpy does not like missing values
                # and will return False whenever some are found in a or b.
                # The proper way would be to use the MaskArray stuff available
                # in Numpy. However, it looks like it has been added to Numpy's
                # core recently, so it may not be available to everyone. Thus,
                # for now we use a home-made recipe, that should probably be
                # revisited in the future.
                a_missing = numpy.isnan(a)
                a_inf = numpy.isinf(a)

                if not (a_missing.any() or (allow_remove_inf and a_inf.any())):
                    # There are no missing values in a, thus this is not the
                    # reason why numpy.allclose(a, b) returned False.
                    _logger.info(
                        'numpy allclose failed for abs_err %f and rel_err %f',
                        numpy.max(abs(a - b)),
                        numpy.max(abs(a - b) / (abs(a) + abs(b))))
                    return False
                # The following line is what numpy.allclose bases its decision
                # upon, according to its documentation.
                rtol = 1.0000000000000001e-05
                atol = 1e-8
                cmp_elemwise = (numpy.absolute(a - b) <=
                        (atol + rtol * numpy.absolute(b)))
                # Find places where both a and b have missing values.
                both_missing = a_missing * numpy.isnan(b)

                # Find places where both a and b have inf of the same sign.
                both_inf = a_inf * numpy.isinf(b)

                # cmp_elemwise is weird when we have inf and -inf.
                # set it to False
                cmp_elemwise = numpy.where(
                        both_inf & cmp_elemwise,
                        a == b,
                        cmp_elemwise)

                # check the sign of the inf
                both_inf = numpy.where(both_inf, (a == b), both_inf)

                if allow_remove_inf:
                    both_inf += a_inf
                if allow_remove_nan:
                    both_missing += a_missing

                # Combine all information.
                return (cmp_elemwise + both_missing + both_inf).all()

        return False

    @staticmethod
    def values_eq_approx_remove_inf(a, b):
        raise NotImplementedError("copy-paste from TensorType")
        return TensorType.values_eq_approx(a, b, True)

    @staticmethod
    def values_eq_approx_remove_nan(a, b):
        raise NotImplementedError("copy-paste from TensorType")
        return TensorType.values_eq_approx(a, b, False, True)

    @staticmethod
    def values_eq_approx_remove_inf_nan(a, b):
        raise NotImplementedError("copy-paste from TensorType")
        return TensorType.values_eq_approx(a, b, True, True)

    def __hash__(self):
        """Hash equal for TensorStorageType with same hash of TensorType"""
        return hashtype(self) ^ hash(self.tensor_type)

    def make_variable(self, name=None):
        """Return a `TensorVariable` of this type

        :Parameters:
         - `name`: str
           A pretty name to identify this `Variable` when printing and
           debugging
        """
        raise NotImplementedError("copy-paste from TensorType")
        return TensorVariable(self, name=name)

    def __str__(self):
        return 'TensorStorageType(%s)' % str(self.tensor_type)

    def __repr__(self):
        raise NotImplementedError("copy-paste from TensorType")
        return str(self)
        #"TensorType{%s, %s}" % (str(self.dtype), str(self.broadcastable))

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
        Variable.__init__(self, type=TensorStorageType(type), name=name,
                                             owner=None, index=None)
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
                type=self.type.tensor_type,
                value=None,
                strict=None,
                storage=self.storage)
        cp.tag = copy.copy(self.tag)
        return cp

    def _as_TensorVariable(self):
        return get_tensor_from_storage(self)

class StoreTensor(Op):
    """
    An Op that stores a numpy ndarray into a TensorStorage.
    (When compiliing on GPU, an the optimizations should change
    the graph such that we end up storing an equivalent cuda
    ndarray to exactly the same TensorStorage).
    Should not be used by the end user; should only be used by
    pfunc.
    """

    view_map = {0: [0]}
    destroy_map = {0: [0]}

    def make_node(self, storage_var, x):

        if not isinstance(x, Variable):
            raise TypeError("Expected x to be a theano Variable.")

        if not isinstance(storage_var, Variable):
            raise TypeError("Expected storage_var to be a variable")

        if not isinstance(storage_var.type, TensorStorageType):
            raise TypeError("Expected storage_var to be a Variable "
                    "of type TensorStorageType.")

        storage = storage_var.storage

        tensor_type = storage.tensor_type

        if x.type != tensor_type:
            err_msg = ('An update must have the same type as the'
                       ' original storage location (storage=%s,'
                       ' storage.tensor_type=%s,'
                       ' update_val=%s, update_val.type=%s).' % (
                           storage,
                           tensor_type,
                           x,
                           x.type))
            err_sug = ('If the difference is related to the broadcast pattern,'
                       ' you can call the'
                       ' tensor.unbroadcast(var, axis_to_unbroadcast[, ...])'
                       ' function to remove broadcastable dimensions.')
            raise TypeError(err_msg, err_sug)

        output = TensorStorageVariable(
                    name=None,
                    type=tensor_type,
                    value=None,
                    strict=None,
                    storage=storage)

        return Apply(self, [storage_var, x], [output])

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hashtype(self)

    def perform(self, node, inp, out):
        storage, val = inp
        out, = out
        storage.set_value(val, borrow=True)
        out[0] = storage

# Not to be used by end user. Should just be used by pfunc
store_tensor = StoreTensor()
