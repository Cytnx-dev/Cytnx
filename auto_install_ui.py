import os,sys

from tkinter import *
import tkinter.font as font
from tkinter import filedialog



class Final(Frame):
   def __init__(self, parent=None, pid=0,side=LEFT, anchor=N,wt=600,ht=400,is_next=True,is_back=True,next_frame=None,back_frame=None,info_txt="",path_frm=None,path_frm2=None,frames=[],fdict=[],prefix_var=None,cytnx_dir=None):
      Frame.__init__(self, parent)

      self.pid = pid
      self.var = IntVar()
      self.path_frm = path_frm
      #self.path_frm2 = path_frm2
      #self.cytnx_dir = cytnx_dir
      self.frames = frames    
      self.fd = fdict
      self.prefix_var = prefix_var

      self.txtvar = StringVar()
      self.lbl = Label(self,textvariable=self.txtvar,anchor='w',justify=LEFT)
      self.lbl['font'] = font.Font(size=16)
      self.set_info_text(info_txt)
      self.lbl.pack(fill="both")
      
      self.termf = Frame(self, height=400, width=600)
      self.termf.pack(fill=BOTH, expand=YES)

      if is_next:
        self.nbut = Button(self,text="[ install ]",command=self._action_nxt)
        self.nbut.pack(side=RIGHT)

      if is_back:
        self.bfram = back_frame
        self.bbut = Button(self,text="<- back",command=self._action_bak)
        self.bbut.pack(side=RIGHT)

   def _analysis(self):
        str_print = "";
    
        if(self.frames[self.fd['MKL']].state_str()=="ON"):
            str_print += "[x] USE_MKL\n"
            str_print += "[x] USE_OMP [force by mkl]\n"
        else:
            str_print += "[ ] USE_MKL\n"
            if(self.frames[self.fd['OMP']].state_str()=="ON"):
                str_print += "[x] USE_OMP\n"

        if(self.frames[self.fd['HPTT']].state_str()=="ON"):
            str_print += "[x] USE_HPTT\n"

            if(self.frames[self.fd['HPTT_finetune']].state_str()=="ON"):
                str_print += "  [x] HPTT finetune\n"
            else:
                str_print += "  [ ] HPTT finetune\n"

            if(self.frames[self.fd['HPTT_arch']].state_str()=="AVX"):
                str_print += "  [x] HPTT optim option = AVX\n"
            elif(self.frames[self.fd['HPTT_arch']].state_str()=="IBM"):
                str_print += "  [x] HPTT optim option = IBM\n"
            elif(self.frames[self.fd['HPTT_arch']].state_str()=="ARM"):
                str_print += "  [x] HPTT optim option = ARM\n"
            else:
                str_print += "  [ ] HPTT optim option\n"

        else:
            str_print += "[ ] USE_HPTT\n"

        if(self.frames[self.fd['CUDA']].state_str()=="ON"):
            str_print += "[x] USE_CUDA\n"
            if(self.frames[self.fd['CUTT']].state_str()=="ON"):
                str_print += "  [x] USE_CUTT\n"
                if(self.frames[self.fd['CUTT_finetune']].state_str()=="ON"):
                    str_print += "    [x] CUTT finetune\n"
                else:
                    str_print += "    [ ] CUTT finetune\n"
            else:
                str_print += "  [ ] USE_CUTT\n"
        else:
            str_print += "[ ] USE_CUDA\n"
                
        if(self.frames[self.fd['PY']].state_str()=="ON"):
            str_print += "[x] BUILD_PYTHON API\n"
        else:
            str_print += "[ ] BUILD_PYTHON API\n"
        
        self.txtvar.set("Review install:\n"+str_print)

   def _action_nxt(self):
        print("Review install")
        ## analysis all flags and generate command.    
        strout = "cmake";
        
        if not  self.prefix_var.get()=='default':
            strout += " -DCMAKE_INSTALL_PREFIX=%s"%(self.prefix_var.get())
    
        if(self.frames[self.fd['MKL']].state_str()=="ON"):
            strout += " -DUSE_MKL=on"
        else:
            if(self.frames[self.fd['OMP']].state_str()=="ON"):
                strout += " -DUSE_OMP=on"

        if(self.frames[self.fd['HPTT']].state_str()=="ON"):
            strout += " -DUSE_HPTT=on"

            if(self.frames[self.fd['HPTT_finetune']].state_str()=="ON"):
                strout += " -DHPTT_ENABLE_FINE_TUNE=on"

            if(self.frames[self.fd['HPTT_arch']].state_str()=="AVX"):
                strout += " -DHPTT_ENABLE_AVX=on"
            elif(self.frames[self.fd['HPTT_arch']].state_str()=="IBM"):
                strout += " -DHPTT_ENABLE_IBM=on"
            elif(self.frames[self.fd['HPTT_arch']].state_str()=="ARM"):
                strout += " -DHPTT_ENABLE_ARM=on"


        if(self.frames[self.fd['CUDA']].state_str()=="ON"):
            strout += " -DUSE_CUDA=on"
            if(self.frames[self.fd['CUTT']].state_str()=="ON"):
                strout += " -DUSE_CUTT=on"
                if(self.frames[self.fd['CUTT_finetune']].state_str()=="ON"):
                    strout += " -DCUTT_ENABLE_FINE_TUNE=on"
                
        if(self.frames[self.fd['PY']].state_str()=="ON"):
            strout += " -DBUILD_PYTHON=on"
        else:
            strout += " -DBUILD_PYTHON=off"
           
        strout += " ../\n" 
        """
        strout += " %s"%(self.cytnx_dir.get())
        ## check:
        if(len(self.cytnx_dir.get())==0):
            raise ValueError("[ERROR] invalid cytnx source path.")
        else:
            if not os.path.exists(self.cytnx_dir.get()):
                raise ValueError("[ERROR] invalid cytnx source path. cannot find path.")
        """
    
        # hide all op buttoms
        self.lbl.destroy()
        self.bbut.destroy()
        self.nbut.destroy()
        self.path_frm.destroy()
        #self.path_frm2.destroy()
        f = open("ainstall.sh",'w')
        f.write("echo $PWD\n")
        f.write("rm -rf build\n")
        f.write("mkdir build\n")
        f.write("cd build\n")
        f.write("echo $PWD\n")
        f.write(strout)
        f.write("make\n")
        f.write("make install")

        os.system('xterm -into %d -geometry 95x30 -s -sb -e sh ainstall.sh&' %(self.termf.winfo_id()))
        #os.system('xterm -into %d -geometry 40x20 -sb -e %s &' %(self.termf.winfo_id(),"cpuinfo"))

   def _action_bak(self):
        self.pack_forget()
        self.bfram.pack(side=TOP,fill=X)
   
   def jump_pack(self,direction,N,start_frame):

        if N>0:
            raise ValueError("cannot call jump on final page.") 
        else:
            self.set_back_frame(start_frame)
            self._analysis()
            self.pack(fill="both")

   def state_id(self):
        return self.var.get()



   def set_back_frame(self,back_frame):
        self.bfram = back_frame

   def set_info_text(self,txt):
        self.txtvar.set(txt)




class Optionbar(Frame):
   def __init__(self, parent=None, pid=0,picks=[], picks_js=None, side=LEFT, anchor=N,wt=600,ht=400,is_next=True,is_back=True,next_frame=None,back_frame=None,info_txt=""):
      Frame.__init__(self, parent)


      self.pid = pid
      self.var = IntVar()
      self.picks = picks
      if picks_js is None:
           self.picks_js = [1 for i in range(len(picks))]
      else:
           self.picks_js = picks_js

      self.dic = dict(zip(picks,range(len(picks))))
      
      self.txtvar = StringVar()
      lbl = Label(self,textvariable=self.txtvar,anchor=W)
      lbl['font'] = font.Font(size=16)
      self.set_info_text(info_txt)
      lbl.pack(fill='both')




      for pick in picks:
         chk = Radiobutton(self, text=pick, variable=self.var,value=self.dic[pick])
         chk.pack(side=side, anchor=anchor, expand=YES)


      if is_next:
        self.nfram = next_frame
        self.nbut = Button(self,text="next ->",command=self._action_nxt)
        self.nbut.pack(side=RIGHT)

      if is_back:
        self.bfram = back_frame
        self.bbut = Button(self,text="<- back",command=self._action_bak)
        self.bbut.pack(side=RIGHT)



   def _action_nxt(self):
        self.pack_forget()
        self.jump_pack('nxt',self.picks_js[self.state_id()],self)

   def _action_bak(self):
        self.pack_forget()
        self.bfram.pack(fill='both')

   def jump_pack(self,direction,N,start_frame):

        if N>0:
            if direction == 'nxt':
                self.nfram.jump_pack('nxt',N-1,start_frame)    
            else:
                raise ValueError("direction should be 'nxt' or 'bak'") 
        else:
            self.set_back_frame(start_frame)
            self.pack(fill='both')
        

   def state_id(self):
        return self.var.get()

   def state_str(self):
        return self.picks[self.var.get()]

   def set_next_frame(self,next_frame):
        self.nfram = next_frame

   def set_back_frame(self,back_frame):
        self.bfram = back_frame

   def set_info_text(self,txt):
        self.txtvar.set(txt)

   def set_default(self,val,by_str=True):
        ival = val
        if by_str:
            ival = self.dic[val]
        self.var.set(ival)

top = Tk()
top.title("Cytnx installer")
#top.geometry("400x300")
top.resizable(False,False)

#main.mainloop()

PREFIX = None
prefix_var = StringVar()
prefix_var.set("default")
def get_prefix():
    PREFIX = filedialog.askdirectory(title = "Select directory to install cytnx")
    prefix_var.set(PREFIX)
    print(PREFIX)
"""
CYTNX_DIR = None
cytnx_var = StringVar()
cytnx_var.set("")
def get_cytnx_dir():
    CYTNX_DIR = filedialog.askdirectory(title = "Select cytnx source path")
    cytnx_var.set(CYTNX_DIR)
    print(CYTNX_DIR)
"""
frm = Frame(top)
pp = Label(frm,text="install path:",anchor=W)
pp.pack(side=LEFT) 
p_str = Label(frm,textvariable=prefix_var,anchor=W)
p_str.pack(side=LEFT)
but_f = Button(frm,text="choose directory to install",command=get_prefix,anchor=E)
but_f.pack(side=RIGHT)
frm.pack(side=TOP,fill=X)
"""
frm2 = Frame(top)
pp2 = Label(frm2,text="cytnx source path:",anchor=W)
pp2.pack(side=LEFT) 
p2_str = Label(frm2,textvariable=cytnx_var,anchor=W)
p2_str.pack(side=LEFT)
but_f = Button(frm2,text="choose cytnx source path",command=get_cytnx_dir,anchor=E)
but_f.pack(side=RIGHT)
frm2.pack(side=TOP,fill=X)
"""

ftype = []

## page mkl
ftype.append("MKL")
mkl_tk = Optionbar(top,0,['ON','OFF'],picks_js=[2,1],is_back=False) 
mkl_tk.set_default('OFF')
mkl_tk.set_info_text("use mkl as linalg library? (default: OFF)\n"+ 
                     "[Note] 1. default use openblas\n"+
                     "[Note] 2. if ON, openmp is forced enable."
                     )

## page omp
ftype.append("OMP")
omp_tk = Optionbar(top,1,['ON','OFF']) 
omp_tk.set_default('OFF')
omp_tk.set_info_text("accelerate using OpenMP? (default: OFF)")


## page hptt
ftype.append("HPTT")
hptt_tk = Optionbar(top,2,['ON','OFF'],picks_js=[1,3]) 
hptt_tk.set_default('OFF')
hptt_tk.set_info_text("accelerate tensor transpose using HPTT lib? (default: OFF)")

ftype.append("HPTT_finetune")
hptt_op2_tk = Optionbar(top,3,['ON','OFF']) 
hptt_op2_tk.set_default('OFF')
hptt_op2_tk.set_info_text("build HPTT lib with optimization on current hardware? (default: OFF)")

ftype.append("HPTT_arch")
hptt_op_tk = Optionbar(top,4,['AVX','IBM','ARM','OFF']) 
hptt_op_tk.set_default('OFF')
hptt_op_tk.set_info_text("build HPTT lib with additional instructions support? (default: OFF)")

## page cuda
ftype.append("CUDA")
cuda_tk = Optionbar(top,5,['ON','OFF'],picks_js=[1,2]) 
cuda_tk.set_default('OFF')
cuda_tk.set_info_text("install GPU(CUDA) support in cytnx? (default: OFF)")

## cutt
ftype.append("CUTT")
cutt_tk = Optionbar(top,6,['ON','OFF'],picks_js=[1,2]) 
cutt_tk.set_default('OFF')
cutt_tk.set_info_text("accelerate tensor transpose on GPU using cuTT lib? (default: OFF)")

ftype.append("CUTT_finetune")
cutt_op_tk = Optionbar(top,7,['ON','OFF']) 
cutt_op_tk.set_default('OFF')
cutt_op_tk.set_info_text("build cuTT lib with optimization on current hardware? (default: OFF)")

## page python 
ftype.append("PY")
python_tk = Optionbar(top,8,['ON','OFF']) 
python_tk.set_default('ON')
python_tk.set_info_text("build python API? (default: ON)")


## final wrapping up
td = dict(zip(ftype,range(len(ftype))))
fin_tk = Final(top,10,path_frm=frm,frames=[mkl_tk,omp_tk,hptt_tk,hptt_op2_tk,hptt_op_tk,cuda_tk,cutt_tk,cutt_op_tk,python_tk],fdict=td,prefix_var=prefix_var)
fin_tk.set_info_text("Review install")


## chain: 
mkl_tk.set_next_frame(omp_tk)
omp_tk.set_next_frame(hptt_tk)
hptt_tk.set_next_frame(hptt_op2_tk)
hptt_op2_tk.set_next_frame(hptt_op_tk)
hptt_op_tk.set_next_frame(cuda_tk)
cuda_tk.set_next_frame(cutt_tk)
cutt_tk.set_next_frame(cutt_op_tk)
cutt_op_tk.set_next_frame(python_tk)
python_tk.set_next_frame(fin_tk)

## visible entry point
mkl_tk.pack(side=TOP,fill="both")



top.mainloop()

exit(1)


    
def bool2str(bl):
    if bl:
        return "ON"
    else:
        return "OFF"



## list all the major options:
USE_MKL=False
USE_OMP=False


USE_CUDA=False
USE_CUTT=False
#CUTT_option_noalign=False
CUTT_option_finetune=False


USE_HPTT=False
HPTT_option_AVX=False
HPTT_option_IBM=False
HPTT_option_ARM=False
HPTT_option_finetune=False


BUILD_PYTHON=True

PREFIX=None


## checking linalg, and openmp.
tmp = input("[2] use mkl as linalg library (default OFF)? (Y/N):")
if(len(tmp.strip())!=0):
    USE_MKL=resolve_yn(tmp)

print("  >>USE_MKL: ",USE_MKL)
print("--------------")
if(USE_MKL):
    print("    -->[2a] force USE_OMP=True")
    print("--------------")
else:
    tmp = input("[2a] use openmp accelerate (default OFF)? (Y/N):")
    if(len(tmp.strip())!=0):
        USE_OMP=resolve_yn(tmp)
    print("  >>USE_OMP:",USE_OMP)
    print("--------------")


## checking HPTT:
tmp = input("[3] use hptt library to accelrate tensor transpose (default OFF)? (Y/N):")
if(len(tmp.strip())!=0):
    USE_HPTT=resolve_yn(tmp)

print("  >>USE_HPTT: ",USE_HPTT)
print("--------------")
if USE_HPTT:
    ## additional options:
    tmp = input("[3a] hptt option(1): fine tune for the native hardware (default OFF)? (Y/N):")
    if(len(tmp.strip())!=0):
        HPTT_option_finetune=resolve_yn(tmp)
    print("  >>HPTT_ENABLE_FINE_TUNE:",HPTT_option_finetune)
    print("--------------")
    
    tmp = input("[3b] hptt option(2): variant options (1: AVX 2: IBM 3: ARM, default OFF)? (1,2,3 or enter for default):")
    if(len(tmp.strip())!=0):
        hptttype=resolve_num(tmp,{1,2,3})
        if(hptttype==1):
            HPTT_option_AVX=True
            print("  >>HPTT_ENABLE_ABX:",HPTT_option_AVX)
        elif(hptttype==2):
            HPTT_option_IBM=True
            print("  >>HPTT_ENABLE_IBM:",HPTT_option_IBM)
        elif(hptttype==3):
            HPTT_option_ARM=True
            print("  >>HPTT_ENABLE_ARM:",HPTT_option_ARM)
        else:
            print("  *No additional options for hptt*")
        print("--------------")

## checking CUDA:
tmp = input("[4] with GPU (CUDA) support (default OFF)? (Y/N):")
if(len(tmp.strip())!=0):
    USE_CUDA=resolve_yn(tmp)

print("  >>USE_CUDA: ",USE_CUDA)
print("--------------")
if USE_CUDA:
    ## additional options:
    tmp = input("[4a] cuda option(1): use cutt library to accelerate tensor transpose (default OFF)? (Y/N):")
    if(len(tmp.strip())!=0):
        USE_CUTT=resolve_yn(tmp)
    print("  >>USE_CUTT:",USE_CUTT)
    print("--------------")
    
    if USE_CUTT:
        ## add-additional options:
        tmp = input("[4a-1] cutt option(1): fine tune for the native hardware (default OFF)? (Y/N):")
        if(len(tmp.strip())!=0):
            CUTT_option_finetune=resolve_yn(tmp)
        print("  >>CUTT_ENABLE_FINE_TUNE:",CUTT_option_finetune)
        print("--------------")
        
        
## checking PYTHON:
tmp = input("[5] Build python API (default ON)? (Y/N):")
if(len(tmp.strip())!=0):
    BUILD_PYTHON=resolve_yn(tmp)

print("  >>BUILD_PYTHON: ",BUILD_PYTHON)
print("--------------")


##=================================================================
print("*************************")
print("  Review install option  ")
print("")

print(" USE_MKL: ",USE_MKL)
print(" USE_OMP: ",USE_OMP)

print(" USE_HPTT: ",USE_HPTT)
if(USE_HPTT):
    print(" -- HPTT_option: ")
    print("    HPTT_FINE_TUNE: ",HPTT_option_finetune)
    if(HPTT_option_AVX):
        print("    HPTT_ENABLE_ABX:",HPTT_option_AVX)
    if(HPTT_option_IBM):
        print("    HPTT_ENABLE_IBM:",HPTT_option_IBM)
    if(HPTT_option_ARM):
        print("    HPTT_ENABLE_ARM:",HPTT_option_ARM)

print(" USE_CUDA: ",USE_CUDA)
print(" USE_CUTT: ",USE_CUTT)
if(USE_CUTT):
    print(" -- CUTT_option: ")
    print("    CUTT_ENABLE_FINE_TUNE: ",CUTT_option_finetune)

print(" BUILD_PYTHON: ",BUILD_PYTHON)
print("*************************")


## generate sh file:
f = open("ainstall.sh",'w')
f.write("rm -rf build\n")
f.write("mkdir build\n")
f.write("cd build\n")
f.write("cmake")
if not PREFIX is None:
    f.write(" -DCMAKE_INSTALL_PREFIX=%s"%(PREFIX))
if(USE_MKL):
    f.write(" -DUSE_MKL=on")
else:
    if(USE_OMP):
        f.write(" -DUSE_OMP=on")

if(USE_HPTT):
    f.write(" -DUSE_HPTT=on")
    if(HPTT_option_finetune):
        f.write(" -DHPTT_ENABLE_FINE_TUNE=on")        

        
    if(HPTT_option_AVX):
        f.write(" -DHPTT_ENABLE_AVX=on")
    if(HPTT_option_IBM):
        f.write(" -DHPTT_ENABLE_IBM=on")
    if(HPTT_option_ARM):
        f.write(" -DHPTT_ENABLE_ARM=on")

if(USE_CUDA):
    f.write(" -DUSE_CUDA=on")
    if(USE_CUTT):
        f.write(" -DUSE_CUTT=on")
        if(CUTT_option_finetune):
            f.write("-DCUTT_ENABLE_FINE_TUNE=on")
        
if(BUILD_PYTHON):
    f.write(" -DBUILD_PYTHON=on")
else:
    f.write(" -DBUILD_PYTHON=off")

f = open("ainstall.sh",'w')
f.write("rm -rf build\n")
f.write("mkdir build\n")
f.write("cd build\n")
f.write("cmake")

f.write(" ../\n")
f.write("make\n")
f.write("make install")

f.close()


