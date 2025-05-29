class MyOp(cytnx.LinOp):
    AddConst = 1# class member.

    def __init__(self,aconst):
        # here, we fix nx=4, dtype=double on CPU,
        # so the constructor only takes the external argument 'aconst'

        ## Remember to init the mother class.
        ## Here, we don't specify custom_f!
        LinOp.__init__(self,"mv",4,cytnx.Type.Double,\
                                   cytnx.Device.cpu )

        self.AddConst = aconst
