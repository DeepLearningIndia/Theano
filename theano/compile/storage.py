"""
Module defining the Storage class.
"""

__authors__ = "Ian Goodfellow"
__copyright__ = "(c) 2013, Universite de Montreal"
__license__ = "3-clause BSD License"
__contact__ = "theano-dev <theano-dev@googlegroups.com>"

class Storage(object):
    """
    An abstract base class representing objects capable of
    storing values between calls to theano functions.
    These Storage objects support multiple views of the same
    kinds of data (for example, the TensorStorage subclass
    provides the same underlying data as either numpy or cuda
    ndarrays).
    """

    def get_value(self, borrow=False):
        """
        Return the value stored in this Storage object.
        Subclasses may need to add parameters to this method to control
        the format of the returned value.

        :param borrow: True to permit returning of an object aliased
            to internal memory.
        """

        raise NotImplementedError(str(type(self)) +
                " does not implement get_value.")


    def set_value(self, value, borrow=False):
        """
        Set the value stored in this Storage object.

        :param borrow:
            True to permit using the value directly, potentially creating problems
            related to aliased memory.

        """

        raise NotImplementedError(str(type(self))+
                " does not implement set_value.")
