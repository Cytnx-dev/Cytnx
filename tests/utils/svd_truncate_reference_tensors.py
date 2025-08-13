import cytnx
# import numpy
src_file_dir = "../test_data_base/common/"

#create random matrix with exponential decaying singular values; U1 version, F64
symms = [cytnx.Symmetry.U1()];
bond_a = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(0)>>3, cytnx.Qs(1)>>2, cytnx.Qs(-1)>>1], symms)
bond_b = cytnx.Bond(cytnx.BD_OUT, [cytnx.Qs(0)>>2, cytnx.Qs(2)>>1, cytnx.Qs(-1)>>2], symms)
bond_c = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(0)>>1, cytnx.Qs(1)>>1, cytnx.Qs(-1)>>1], symms)
bond_d = cytnx.Bond(cytnx.BD_OUT, [cytnx.Qs(2)>>2, cytnx.Qs(1)>>1, cytnx.Qs(-1)>>1, cytnx.Qs(0)>>2], symms)
Torigin = cytnx.UniTensor([bond_a, bond_b, bond_c, bond_d],dtype=cytnx.Type.Double,
                        name="U1 UniTensor (random)",
                        labels=["a","b","c","d"])
Torigin.normal_(0.2, 1., 1000)
[S, Uex, vTex] = cytnx.linalg.Gesvd(Torigin, True, True);
singvals=min(S.shape())
# randperm = numpy.random.permutation(singvals) # random permutation, such that the singular values are not ordered
randperm = [11, 13, 12,  0, 14,  9,  7,  4, 10,  8,  5,  6,  3,  2, 15, 1]
for k in range(singvals):
  expon = -16*k/singvals
  S[randperm[k],randperm[k]] = pow(10,expon)
Texp=cytnx.Contract(Uex, S)
Texp=cytnx.Contract(Texp, vTex)
Texp.set_name("U(1) UniTensor with exponentially falling singular values in the range 1 to 10^-16")
Texp.Save(src_file_dir + "sym_UT_U1_exp_Svals_F64")

#calculate truncated singular values and reconstructed matrix for truncated SVD; F64 version
src_file_name = "sym_UT_U1_exp_Svals_F64"
res_file_dir = "../test_data_base/linalg/Svd_truncate/"
src_T = cytnx.UniTensor.Load(src_file_dir + src_file_name + ".cytnx");
keepdim = 5
[Sref, Uref, vTref] = cytnx.linalg.Gesvd_truncate(src_T,keepdim);
Mref = cytnx.Contract(Uref, Sref)
Mref = cytnx.Contract(Mref, vTref)
Mref.make_contiguous()
Sref.Save(res_file_dir + src_file_name)
Mref.Save(res_file_dir + src_file_name + "_reconstructed")

#calculate truncated singular values and reconstructed matrix for truncated SVD with minblockdim set; F64 version
src_file_name = "sym_UT_U1_exp_Svals_F64"
res_file_dir = "../test_data_base/linalg/Svd_truncate/"
src_T = cytnx.UniTensor.Load(src_file_dir + src_file_name + ".cytnx");
keepdim = 5
[Sref, Uref, vTref] = cytnx.linalg.Gesvd_truncate(src_T,keepdim=keepdim,min_blockdim=[1, 1, 0, 2, 0]);
Mref = cytnx.Contract(Uref, Sref)
Mref = cytnx.Contract(Mref, vTref)
Mref.make_contiguous();
Sref.Save(res_file_dir + src_file_name + "_minblockdim")
Mref.Save(res_file_dir + src_file_name + "_minblockdim_reconstructed")

#create random matrix with exponential decaying singular values; U1 version, C128
symms = [cytnx.Symmetry.U1()];
bond_a = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(0)>>3, cytnx.Qs(1)>>2, cytnx.Qs(-1)>>1], symms)
bond_b = cytnx.Bond(cytnx.BD_OUT, [cytnx.Qs(0)>>2, cytnx.Qs(2)>>1, cytnx.Qs(-1)>>2], symms)
bond_c = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(0)>>1, cytnx.Qs(1)>>1, cytnx.Qs(-1)>>1], symms)
bond_d = cytnx.Bond(cytnx.BD_OUT, [cytnx.Qs(2)>>2, cytnx.Qs(1)>>1, cytnx.Qs(-1)>>1, cytnx.Qs(0)>>2], symms)
Torigin = cytnx.UniTensor([bond_a, bond_b, bond_c, bond_d],dtype=cytnx.Type.ComplexDouble,
                        name="U1 UniTensor (random)",
                        labels=["a","b","c","d"])
Torigin.normal_(0.2, 1., 1000)
[S, Uex, vTex] = cytnx.linalg.Gesvd(Torigin, True, True);
singvals=min(S.shape())
# randperm = numpy.random.permutation(singvals) # random permutation, such that the singular values are not ordered
randperm = [11, 13, 12,  0, 14,  9,  7,  4, 10,  8,  5,  6,  3,  2, 15, 1]
for k in range(singvals):
  expon = -16*k/singvals
  S[randperm[k],randperm[k]] = pow(10,expon)
Texp=cytnx.Contract(Uex, S)
Texp=cytnx.Contract(Texp, vTex)
Texp.set_name("U(1) UniTensor with exponentially falling singular values in the range 1 to 10^-16")
Texp.Save(src_file_dir + "sym_UT_U1_exp_Svals_C128")

#calculate truncated singular values and reconstructed matrix for truncated SVD; C128 version
src_file_name = "sym_UT_U1_exp_Svals_C128"
res_file_dir = "../test_data_base/linalg/Svd_truncate/"
src_T = cytnx.UniTensor.Load(src_file_dir + src_file_name + ".cytnx");
keepdim = 5
[Sref, Uref, vTref] = cytnx.linalg.Gesvd_truncate(src_T,keepdim);
Mref = cytnx.Contract(Uref, Sref)
Mref = cytnx.Contract(Mref, vTref)
Mref.make_contiguous()
Sref.Save(res_file_dir + src_file_name)
Mref.Save(res_file_dir + src_file_name + "_reconstructed")

#calculate truncated singular values and reconstructed matrix for truncated SVD with minblockdim set; C128 version
src_file_name = "sym_UT_U1_exp_Svals_C128"
res_file_dir = "../test_data_base/linalg/Svd_truncate/"
src_T = cytnx.UniTensor.Load(src_file_dir + src_file_name + ".cytnx");
keepdim = 5
[Sref, Uref, vTref] = cytnx.linalg.Gesvd_truncate(src_T,keepdim=keepdim,min_blockdim=[1, 1, 0, 2, 0]);
Mref = cytnx.Contract(Uref, Sref)
Mref = cytnx.Contract(Mref, vTref)
Mref.make_contiguous();
Sref.Save(res_file_dir + src_file_name + "_minblockdim")
Mref.Save(res_file_dir + src_file_name + "_minblockdim_reconstructed")
