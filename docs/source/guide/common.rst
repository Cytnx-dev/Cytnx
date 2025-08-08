Common APIs
=============

The central objects that store data in Cytnx such as **Storage**, **Tensor** and **UniTensor** share very similar member functions and attributes, in this seciton we summerize these APIs and the hyperlink to the correspnding guide page for convenience.


Data type
************

.. py:function:: .dtype()

    The dtype-id of current Storage, Tensor or UniTensor.

.. py:function:: .dtype_str()

 	The dtype (string) of current Storage, Tensor or UniTensor.

.. py:function:: .astype(type)

    Cast the type of current Storage, Tensor or UniTensor.

.. py:function:: .real()

    Get the real part of the current Storage or Tensor.

.. py:function:: .imag()

    Get the imaginary part of the current Storage or Tensor.

See:
    :ref:`Creating a Storage`

    :ref:`Creating a Tensor`

    :ref:`Creating a UniTensor`

Device
************

.. py:function:: .device()

    The device-id of current Storage, Tensor or UniTensor.

.. py:function:: .device_str()

 	The device(string) of current Storage, Tensor or UniTensor.

.. py:function:: .to(device)

	move the current Storage, Tensor or UniTensor to different deivce.

.. py:function:: .to_(device)

    move a new Storage, Tensor or UniTensor with same content as current object on different deivce.

See:
    :ref:`Creating a Storage`

    :ref:`Creating a Tensor`

    :ref:`Creating a UniTensor`

Save & Load
************

.. py:function:: .Save(name)

    Save current Storage, Tensor or UniTensor to file.

.. py:function:: .Load(name)

 	Load a Storage, Tensor or UniTensor from file.

.. py:function:: .Tofile(name)

    Save current Storage or Tensor to binary file without any additional header information.

.. py:function:: .Fromfile(name)

 	Load the Storage or Tensor from binary file.

See:
    :ref:`Save/Load a Storage`

    :ref:`Save/Load a Tensor`

    :ref:`Save/Load a UniTensor`

Operations
************

.. py:function:: .clone()

    Return a clone of the current Storage, Tensor or UniTensor.
