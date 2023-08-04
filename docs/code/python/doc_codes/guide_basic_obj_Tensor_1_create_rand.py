A = cytnx.random.normal([3,4,5], mean=0., std=1.)   #Tensor of shape (3,4,5) with all elements distributed according
                                                    #to a normal distribution around 0 with standard deviation 1
B = cytnx.random.uniform([3,4,5], low=-1., high=1.) #Tensor of shape (3,4,5) with all elements distributed uniformly
                                                    #between -1 and 1
