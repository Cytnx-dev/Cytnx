from graphviz import Graph, Digraph


def Drawnet_notag(opt_name,tn_names, edges_info,dangling_infos,figsize=[6,5],engine='circo',format="pdf"):
    ## edges_info[i] = (tn1_name,tn2_name,common_label)

    g = Graph(opt_name,filename=opt_name+".gv",engine=engine,format=format)
    #g = Graph(engine=engine,format=format)
    g.attr(size='%d,%d'%(figsize[0],figsize[1]))
    ## insert node!
    g.attr('node',shape='circle')
    for i in tn_names:
        g.node(i)


    g.attr('node',shape='plaintext')
    for i in dangling_infos:
        g.node(i[0])
    #print(edges_info)
    ## edges! contracted:
    for i in edges_info:
        g.edge(i[0],i[1],label="%d"%(i[2]))


    ## edges! non-contracted:
    for i in dangling_infos:
        g.edge(i[0],i[1])

    g.view()
