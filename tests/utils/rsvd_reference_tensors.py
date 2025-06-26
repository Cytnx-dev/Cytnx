import cytnx
src_file_dir = "../test_data_base/common/"

#create random matrix with exponential decaying singular values; F64 version
singvals=20;
Torigin = cytnx.random.uniform([singvals,2*singvals], -1., 1., dtype=cytnx.Type.Float, seed=10000);
[Sex, Uex, vTex] = cytnx.linalg.Gesvd(Torigin, True, True);
Snew=cytnx.Tensor([Uex.shape()[1], vTex.shape()[0]]);
for k in range(0,singvals):
  expon = -16*k/singvals;
  Snew[k,k] = pow(10,expon);
Texp=cytnx.linalg.Matmul(Uex, Snew)
Texp=cytnx.linalg.Matmul(Texp, vTex)
uB = cytnx.UniTensor(Texp.reshape_(20, 4, 10), labels=["one", 'two', 'three']).set_name("Dense UniTensor with exponentially falling singular values in the range 1 to 10^-16")
uB.Save(src_file_dir + "dense_nondiag_exp_Svals_F64")

#calculate truncated singular values and reconstructed matrix for Rsvd_truncate; F64 version
src_file_name = "dense_nondiag_exp_Svals_F64"
res_file_dir = "../test_data_base/linalg/Svd_truncate/"
src_T = cytnx.UniTensor.Load(src_file_dir + src_file_name + ".cytnx");
keepdim = 5
[Sref, Uref, vTref] = cytnx.linalg.Gesvd_truncate(src_T,keepdim);
Mref = cytnx.Contract(Uref, Sref)
Mref = cytnx.Contract(Mref, vTref)
Mref.make_contiguous()
Sref.Save(res_file_dir + src_file_name)
Mref.Save(res_file_dir + src_file_name + "_reconstructed")

#calculate truncated singular values for Rsvd; F64 version
src_file_name = "dense_nondiag_exp_Svals_F64"
res_file_dir = "../test_data_base/linalg/Rsvd/"
src_T = cytnx.UniTensor.Load(src_file_dir + src_file_name + ".cytnx");
keepdim = 15
[Sref, Uref, vTref] = cytnx.linalg.Gesvd_truncate(src_T,keepdim);
Sref.Save(res_file_dir + src_file_name)

#create random matrix with exponential decaying singular values; C128 version
singvals=20;
Torigin = cytnx.random.uniform([singvals,2*singvals], -1., 1., dtype=cytnx.Type.ComplexDouble, seed=10000);
[Sex, Uex, vTex] = cytnx.linalg.Gesvd(Torigin, True, True);
Snew=cytnx.Tensor([Uex.shape()[1], vTex.shape()[0]]);
for k in range(0,singvals):
  expon = -16*k/singvals;
  Snew[k,k] = pow(10,expon);
Texp=cytnx.linalg.Matmul(Uex, Snew)
Texp=cytnx.linalg.Matmul(Texp, vTex)
uB = cytnx.UniTensor(Texp.reshape_(20, 4, 10), labels=["one", 'two', 'three']).set_name("Dense UniTensor with exponentially falling singular values in the range 1 to 10^-16")
uB.Save(src_file_dir + "dense_nondiag_exp_Svals_C128")

#calculate truncated singular values and reconstructed matrix after truncated SVD; C128 version
src_file_name = "dense_nondiag_exp_Svals_C128"
res_file_dir = "../test_data_base/linalg/Svd_truncate/"
src_T = cytnx.UniTensor.Load(src_file_dir + src_file_name + ".cytnx");
keepdim = 5
[Sref, Uref, vTref] = cytnx.linalg.Gesvd_truncate(src_T,keepdim);
Mref = cytnx.Contract(Uref, Sref)
Mref = cytnx.Contract(Mref, vTref)
Mref.make_contiguous()
Sref.Save(res_file_dir + src_file_name)
Mref.Save(res_file_dir + src_file_name + "_reconstructed")

#calculate truncated singular values for Rsvd; C128 version
src_file_name = "dense_nondiag_exp_Svals_C128"
res_file_dir = "../test_data_base/linalg/Rsvd/"
src_T = cytnx.UniTensor.Load(src_file_dir + src_file_name + ".cytnx");
keepdim = 15
[Sref, Uref, vTref] = cytnx.linalg.Gesvd_truncate(src_T,keepdim);
Sref.Save(res_file_dir + src_file_name)