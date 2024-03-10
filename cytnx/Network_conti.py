from .utils import *
from cytnx import *

# import imp
# try:
#     imp.find_module('graphviz')
#     from .NetGraph import *
# except ImportError:
#     from .NetGraph_empty import *

import importlib
spec = importlib.util.find_spec("graphviz")
if spec is None:
    # print("Can't find the graphviz module.")
    from .NetGraph_empty import *
else:
    # If you chose to perform the actual import ...
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Adding the module to sys.modules is optional.
    #sys.modules["graphviz"] = module
    from .NetGraph import *

import numpy as np


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
            dangling_edges.append(("%d"%(rtnl[i][j]),tn_names[i]))

    return Drawnet_notag(OUT_fn,tn_names,edge_info,dangling_edges,figsize,engine=engine)
