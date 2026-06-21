from .utils import *
from cytnx import *

try:
    from graphviz import Graph
except ImportError:
    Graph = None

import numpy as np


def Drawnet_notag(opt_name,tn_names, edges_info,dangling_infos,figsize=[6,5],engine='circo',format="pdf"):
    if Graph is None:
        raise ModuleNotFoundError("[ERROR] graphviz is not installed!")
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
        g.edge(i[0],i[1],label="%s"%(i[2]))


    ## edges! non-contracted:
    for i in dangling_infos:
        g.edge(i[0],i[1])

    g.view()


@add_method(Network)
def Diagram(self,outname=None,figsize=[6,5],engine="circo"):
    if(self.isLoad()==False):
        print("[ERROR][Network] The Network does not Load any Network file yet.")
        exit(99)


    tn_names = np.array(self._cget_tn_names());
    tn_labels = np.array(self._cget_tn_labels());
    tn_out_lbls = np.array(self._cget_tn_out_labels());
    #print(tn_names)
    #print(tn_labels)
    #print(tn_out_lbls)

    ## get name:
    if(outname is None):
        OUT_fn = self._cget_filename();
    else:
        OUT_fn = outname;
    #print(OUT_fn)

    ## get common label:
    all_l = np.concatenate(tn_labels)
    comm_lbl = np.setdiff1d(all_l,tn_out_lbls)

    #print(all_l)
    #print(comm_lbl)
    edge_info = []
    for i in comm_lbl:
        out = []
        for j in range(len(tn_labels)):
            if i in tn_labels[j]:
                out.append(tn_names[j])
                if(len(out)==2):
                    break;
        out.append(i)
        edge_info.append(tuple(out))

    ## remove the common_label
    rtnl = []
    for j in range(len(tn_labels)):
        rtnl.append(np.setdiff1d(tn_labels[j],comm_lbl))


    dangling_edges = []
    for i in range(len(rtnl)):
        for j in range(len(rtnl[i])):
            dangling_edges.append(("%s"%(rtnl[i][j]),tn_names[i]))

    return Drawnet_notag(OUT_fn,tn_names,edge_info,dangling_edges,figsize,engine=engine)
