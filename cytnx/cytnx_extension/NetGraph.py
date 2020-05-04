from graphviz import Graph, Digraph



def Drawnet_notag(opt_name,tn_names, edges_info,dangling_infos):
    ## edges_info[i] = (tn1_name,tn2_name,common_label)

    g = Graph(opt_name,filename=opt_name+".gv")

    ## insert node!
    g.attr('node',shape='circle')
    for i in tn_names:
        g.node(i)


    g.attr('node',shape='plaintext')
    for i in dangling_infos:
        g.node(i[0])
    
    ## edges! contracted:
    for i in edges_info:
        g.edge(i[0],i[1],label="%d"%(i[2]))


    ## edges! non-contracted:
    for i in dangling_infos:
        g.edge(i[0],i[1])

    g.view()
    





