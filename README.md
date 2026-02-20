# Objectives

The learning objectives of this assignment are to:
1. practice Python programming skills and use of numpy arrays
2. Understand the importance of testing your codes


# Setup your environment as given in the instructions



Please go through some tutorials for your references:

* [Python tutorial](https://docs.python.org/3/tutorial/)
* [numpy tutorial](https://docs.scipy.org/doc/numpy/user/quickstart.html)


You are now ready to begin working on the assignment. Kindly follow the instructions.


# Write your code

You will be editing the file `stv_nn.ipynb` and downloading it as an 'stv_nn.py' file for testing.
**Do not add or edit any other files. Donot change the provided names.**

You will implement an `Index` that associates objects with integer indexes.
This is a very common setup step in training neural networks, which require that
everything be expressed as numbers, not objects.

A template for the `Index` class has been provided to you in the file `stv_nn.ipynb`.
In the template, each method has only a documentation string, with no code in
the body of the method yet.
For example, the `objects_to_indexes` method looks like:
```python
def objects_to_indexes(self, object_seq: Sequence[Any]) -> np.ndarray:
    """
    Returns a vector of the indexes associated with the input objects.

    For objects not in the vocabulary, `start-1` is used as the index.

    :param object_seq: A sequence of objects.
    :return: A 1-dimensional array of the object indexes.
    """
```

You should read the documentation strings (docstrings) in each of methods in
`stv_nn.ipynb`, and implement the methods as described.
Write your code below the docstring of each method;
**do not delete or edit the docstrings**.

The following objects and functions may come in handy:
* [dict](https://docs.python.org/3/library/stdtypes.html#mapping-types-dict)
* [numpy.array](https://numpy.org/doc/stable/reference/generated/numpy.array.html)
* [numpy.full](https://numpy.org/doc/stable/reference/generated/numpy.full.html)
* [numpy.zeros](https://numpy.org/doc/stable/reference/generated/numpy.zeros.html)
* [numpy.stack](https://numpy.org/doc/stable/reference/generated/numpy.stack.html)
* [numpy.nonzero](https://numpy.org/doc/stable/reference/generated/numpy.nonzero.html)

# Test your code for correctness

The `test_stv_nn.py` file contains tests that check that each of the methods of
`Index` behaves as expected.
For example, the `test_indexes` test method looks like:

```python

def test_indexes():
    vocab = ["four", "three", "", "two", "one"]
    objects = ["one", "", "four", "four"]
    indexes = np.array([4, 2, 0, 0])

    index = nn.Index(vocab)
    assert_array_equal(index.objects_to_indexes(objects), indexes)
    assert index.indexes_to_objects(indexes) == objects

    index = nn.Index(vocab, start=1)
    assert_array_equal(index.objects_to_indexes(objects), indexes + 1)
    assert index.indexes_to_objects(indexes + 1) == objects
```
This tests that your code correctly associates indexes with an input vocabulary
``"four", "three", "", "two", "one"``, that it can convert back and forth
between objects and indexes, and that it can handle indexing that starts from a
number other than 0.

To run all the provided tests, run ``pytest`` from the directory containing
``test_nn.py``.
Initially, you will see output like:
```
============================= test session starts ==============================
...
collected 8 items

test_nn.py FFFFFFFF                                                      [100%]

=================================== FAILURES ===================================
_________________________________ test_indexes _________________________________

    @pytest.mark.timeout(1)
    def test_indexes():
        vocab = ["four", "three", "", "two", "one"]
        objects = ["one", "", "four", "four"]
        indexes = np.array([4, 2, 0, 0])

        index = nn.Index(vocab)
>       assert_array_equal(index.objects_to_indexes(objects), indexes)

test_nn.py:21:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

actual = None, desired = array([4, 2, 0, 0])

    def assert_array_equal(actual, desired):
>       assert type(actual) is np.ndarray
E       AssertionError: assert <class 'NoneType'> is <class 'numpy.ndarray'>
E        +  where <class 'NoneType'> = type(None)
E        +  and   <class 'numpy.ndarray'> = np.ndarray

test_nn.py:10: AssertionError
...
============================== 8 failed in 0.73s ===============================
```
This indicates that all tests are failing, which is expected since you have not
yet written the code for any of the methods.
Once you have written the code for all methods, you should instead see
something like:
```
============================= test session starts ==============================
...
collected 8 items

test_nn.py ........                                                      [100%]

============================== 8 passed in 1.49s ===============================
```

# Test your code for quality




# Grading
 * Your work will only graded based on the codes you have submitted in "stv_nn.ipynb" 
 * While there are no penalties for using "pytest," it's evident that when your code is accurate, all your tests will be successful.
 * Testing is considered a best practice in industries to ensure software reliability, identify issues early, and deliver high-quality products to users.