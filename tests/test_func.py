def doc():
    """
    >>> import numpy as np
    >>> a = np.arange(15).reshape(3, 5)
    >>> a
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])
    >>> a.shape
    (3, 5)
    >>> a.ndim
    2
    >>> a.size
    15
    >>> a.dtype.name
    'int32'
    >>> x = 0.00000001
    >>> x
    0.00000001
    >>> x = 0.00000001
    >>> x
    1e-08
    >>> x = 0.0001
    >>> x
    0.0001
    """
    pass


def check_atol():
    """
    >> x = 0.00000001
    >> x
    2e-08
    >>> x = 0.0001
    >>> x
    0.0002
    """
    pass


def doc2():
    """
    >>> import numpy as np
    >>> a = np.array([[3, 2, 0], [1, -1, 0], [0, 5, 1]])
    >>> b = np.array([2, 4, -1])
    >>> from scipy import linalg
    >>> x = linalg.solve(a, b)
    >>> x
    array([ 2., -2.,  9.])
    >>> np.dot(a, x) == b
    array([ True,  True,  True], dtype=bool)
    """
    pass
