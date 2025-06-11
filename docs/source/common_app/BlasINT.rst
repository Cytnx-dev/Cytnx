Check current Blas/Lapack integer size
--------------------------------------------------------------

To check whether the current Blas/Lapack integer size is 4 bytes or 8 bytes:

* In Python:

.. code-block:: python
    :linenos:

    print(cytnx.__blasINTsize__)


* In C++:

.. code-block:: c++
    :linenos:

    print(cytnx::__blasINTsize__);


**Output (example of size of 8bytes) >>**

.. code-block:: text

   8
