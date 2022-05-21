Function Naming convention
============================
Generally, the function naming scheme in Cytnx follows the rule:


1. If the function is **acting on objects** (taking object as arguments), they will start with first Letter being **capical**. Example such as linalg function, contract etc...

    .. code-block:: python
        
        cytnx.linalg.Svd(A)
        cytnx.linalg.Qr(A)
        cytnx.linalg.Sum(A)

        Contract(A,B)


2. If a function is a **member function**, or a **generating function** (such as zeros(), ones() ...), then they usually start with **lower** letter, for example:

    .. code-block:: python

        A = zeros([2,3,4])

        A.permute(0,2,1)
        B = A.contiguous()


3. **Objects** in Cytnx always start with **capical** Letter, for example:

    .. code-block:: python

        A = Tensor()
        B = Bond()
        C = Network()
        C = UniTensor()


4. Functions end with **underscore** means the *input* will be changed. If it is a member function, it is inplace operation


    .. code-block:: python

        A = zeros([2,3,4])

        A.contiguous_() # A gets changed
        B = A.contiguous() # A is not changed, but return a copy B (see Tensor for further info)
        
        A.permute_(0,2,1) # A gets changed
        C = A.permute(0,2,1) # A is not changed but return a new B as A's permute






    






.. toctree::

