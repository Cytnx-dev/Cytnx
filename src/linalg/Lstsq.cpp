#include "linalg.hpp"
#include "linalg_internal_interface.hpp"
#include <iostream>
#include "Tensor.hpp"
#include "Generator.hpp"

namespace cytnx{
    namespace linalg{
        std::vector<Tensor> Lstsq(const Tensor &A,const Tensor &b, const float &rcond) {
            cytnx_error_msg(A.shape().size() != 2 || b.shape().size() != 2, "[Lsq] error, Lstsq can only operate on rank-2 Tensor.%s", "\n");

            cytnx_int64 m = A.shape()[0];
            cytnx_int64 n = A.shape()[1];
            cytnx_int64 nrhs = b.shape()[1];

            Tensor Ain; Tensor bin;
            if(A.is_contiguous()) Ain = A.clone();
            else Ain = A.contiguous();
            if(b.is_contiguous()) bin = b.clone();
            else bin = b.contiguous();

            int type_ = A.dtype()<b.dtype()?A.dtype():b.dtype();
            if(type_ > Type.Float || type_ == Type.Double){
                Ain = Ain.astype(Type.Double);
                bin = bin.astype(Type.Double);
            }else if(type_ == Type.Float){
                Ain = Ain.astype(Type.Float);
                bin = bin.astype(Type.Float);
            }else if(type_ == Type.ComplexFloat){
                Ain = Ain.astype(Type.ComplexFloat);
                bin = bin.astype(Type.ComplexFloat);
            }else if(type_ == Type.ComplexDouble){
                Ain = Ain.astype(Type.ComplexDouble);
                bin = bin.astype(Type.ComplexDouble);
            }

            if(m<n) {
                Storage bstor = bin.storage();
                bstor.resize(n*nrhs);
                bin = Tensor::from_storage(bstor).reshape({n,nrhs});
            }

            std::vector<Tensor> out;
            cytnx_int64* rank = (cytnx_int64*)malloc(sizeof(cytnx_int64));

            Tensor s; s.Init({m<n?m:n}, Ain.dtype()<=2?Ain.dtype()+2:Ain.dtype(), Ain.device());
            s.storage().set_zeros();

            if (A.device() == Device.cpu && b.device() == Device.cpu) {

                cytnx::linalg_internal::lii.Lstsq_ii[Ain.dtype()](Ain._impl->storage()._impl,
                                                                  bin._impl->storage()._impl,
                                                                  s._impl->storage()._impl,
                                                                  m,n,nrhs, rcond, rank);

                Tensor sol = bin(Accessor::range(0,n,1),":");
                sol.reshape_({n,nrhs});
                out.push_back(sol);

                Tensor res = zeros({1});
                if(m>n && rank[0]>=n){
                    Tensor res_ = bin(Accessor::range(n,m,1),":");
                    Tensor ones_ = ones({m-n,1}).reshape({1,m-n});
                    res = linalg::Tensordot(ones_, res_.Pow(2), {1}, {0}, 0, 0);
                }
                out.push_back(res);

                Tensor r = zeros({1}, Type.Int64, Ain.device()); r(0)=rank[0];
                out.push_back(r);
                out.push_back(s);
                return out;

            } else {
#ifdef UNI_GPU
                cytnx_error_msg(true,"[ERROR] currently Lstsq for non-symmetric matrix is not supported.%s","\n");
                return std::vector<Tensor>();
#else
                cytnx_error_msg(true, "[Lsq] fatal error,%s", "try to call the gpu section without CUDA support.\n");
                return std::vector<Tensor>();
#endif
            }
        }
    }
}
