from cytnx import *
from bokeh.io import show
from bokeh.plotting import figure
from bokeh.layouts import row,column
from bokeh.models import Arrow,NormalHead,TeeHead, Range1d
from bokeh.models import Div
import numpy as np

def MakeBond(Bd, yloc, arrow_length, is_left_side, UBody_width):

    if is_left_side:
        if (Bd.type() == BD_IN):
            return Arrow(end=NormalHead(fill_color="orange"),
                   x_start=-arrow_length, y_start=yloc, x_end=0, y_end=yloc)
        elif (Bd.type() == BD_OUT):
            return Arrow(end=NormalHead(fill_color="orange"),
                   x_start=0, y_start=yloc, x_end=-arrow_length, y_end=yloc)
        else:
            return Arrow(end=TeeHead(),
                   x_start=-arrow_length,x_end=0 ,y_start=yloc, y_end=yloc)

    else:
        if (Bd.type() == BD_IN):
            return Arrow(end=NormalHead(fill_color="orange"),
                   x_start=UBody_width+arrow_length, y_start=yloc, x_end=UBody_width, y_end=yloc)
        elif (Bd.type() == BD_OUT):
            return Arrow(end=NormalHead(fill_color="orange"),
                   x_start=UBody_width, y_start=yloc, x_end=UBody_width+arrow_length, y_end=yloc)
        else:
            return Arrow(end=TeeHead(),
                   x_start=UBody_width+arrow_length ,x_end=UBody_width ,y_start=yloc, y_end=yloc)




def _inspect_blocks(UTen):

    # Get essential info:
    Nin = UTen.rowrank()
    Nout = UTen.rank() - UTen.rowrank()

    # Name:
    TN_name = "UniTensor"
    if UTen.name() != "":
        TN_name = UTen.name()

    # type:
    TN_name += f" ({UTen.uten_type_str()})"


    Ht = np.max([Nout,Nin])
    Wt = 2

    arrow_length=1

    p = figure(width=400,height=400)
    p.patches([[0,Wt,Wt,0]], [[0.5,0.5,-Ht+0.5,-Ht+0.5]], color = ["navy"], alpha=[0.3], line_width=2)

    for b in range(Nin):
        p.add_layout(MakeBond(UTen.bond(b),-b,arrow_length,True,Wt))

    for b in range(Nout):
        p.add_layout(MakeBond(UTen.bond(UTen.rowrank()+b),-b,arrow_length,False,Wt))


    p.x_range = Range1d(-arrow_length-1,arrow_length+Wt+1)
    p.y_range = Range1d(-Ht-1.5,1.5)
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    #hiding ticks:
    p.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
    p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
    p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks

    #hiding tick_label:
    p.xaxis.major_label_text_color= None
    p.yaxis.major_label_text_color= None

    title_pan = Div(text="<br>%s</br>"%(TN_name),width=p.width)


    show(column(title_pan,p))


if __name__=="__main__":

    f = UniTensor(zeros([4,5,2,3,4]),rowrank=2)
    _inspect_blocks(f)
