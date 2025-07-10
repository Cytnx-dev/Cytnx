Function Naming convention
============================
Generally, the function naming scheme in Cytnx follows the rules:


1. If the function is **acting on objects** (taking object as arguments), they will start with the first letter being **capical**. Examples are the linalg functions, Contract etc...

    .. code-block:: python

        cytnx.linalg.Svd(A)
        cytnx.linalg.Qr(A)
        cytnx.linalg.Sum(A)

        cytnx.Contract(A,B)


2. If a function is a **member function**, or a **generating function** (such as zeros(), ones() ...), then they usually start with a **lower** case letter, for example:

    .. code-block:: python

        A = cytnx.zeros([2,3,4])

        A.permute(0,2,1)
        B = A.contiguous()


3. **Objects** in Cytnx always start with **capical** letters, for example:

    .. code-block:: python

        A = cytnx.Tensor()
        B = cytnx.Bond()
        C = cytnx.Network()
        C = cytnx.UniTensor()


4. Functions end with **underscore** indicate that the *input* will be changed. For member functions, this is an inplace operation


    .. code-block:: python

        A = cytnx.zeros([2,3,4])

        A.contiguous_() # A gets changed
        B = A.contiguous() # A is not changed, but return a copy B (see Tensor for further info)

        A.permute_(0,2,1) # A gets changed
        C = A.permute(0,2,1) # A is not changed but return a new B as A's permute













.. toctree::
