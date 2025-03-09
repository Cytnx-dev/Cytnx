for i in [0,1]:
    tmp = Tsymm.at([0,0,i])
    if(tmp.exists()):
        tmp.value = 8.
