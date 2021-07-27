#include "tn_algo/MPS.hpp"
#include "random.hpp"
#include <cmath>
#include <algorithm>
#include "linalg.hpp"
using namespace std;
namespace cytnx{
    namespace tn_algo{

        std::ostream& RegularMPS::Print(std::ostream &os){
            os << "MPS type : " << "[Regular]" << endl;
            os << "Size : " << this->_TNs.size() << endl;
            os << "Sloc : " << this->S_loc << endl;
            os << "physBD dim : " << this->phys_dim << endl;
            os << "virtBD dim : " << this->virt_dim << endl;
            os << endl;
            return os;
        }


        void RegularMPS::Init(const cytnx_uint64 &N, const cytnx_uint64 &phys_dim, const cytnx_uint64 &virt_dim){
            //checking:
            cytnx_error_msg(N==0,"[ERROR][RegularMPS] number of site N cannot be ZERO.%s","\n");

            this->phys_dim = phys_dim;
            this->virt_dim = virt_dim;            

            const cytnx_uint64& d = phys_dim; 
            const cytnx_uint64& chi = virt_dim;
            
            this->_TNs.resize(N);
            this->_TNs[0] = UniTensor(cytnx::random::normal({1, d, min(chi, d)}, 0., 1.),2);
            cytnx_uint64 dim1,dim2,dim3;

            for(cytnx_int64 k=1; k<N; k++){
                dim1 = this->_TNs[k-1].shape()[2]; dim2 = d;
                dim3 = std::min(std::min(d, cytnx_uint64(this->_TNs[k-1].shape()[2] * d)), cytnx_uint64(pow(d,N - k - 1)));
                this->_TNs[k] = UniTensor(random::normal({dim1, dim2, dim3},0.,1.),2);
                this->_TNs[k].set_labels({2*k,2*k+1,2*k+2});
            }
            this->S_loc = -1;

            
        }


        void RegularMPS::Into_Lortho(){
            if(this->S_loc == -1){
                this->S_loc = 0;
            }else if(this->S_loc == this->_TNs.size()){
                return;
            }

            for(cytnx_int64 p=0; p<this->size() -1 - this->S_loc;p++){
                auto out = linalg::Svd(this->_TNs[p]);
                auto s = out[0];
                this->_TNs[p] = out[1];
                auto vt = out[2];
                this->_TNs[p+1] = Contract(Contract(s,vt),this->_TNs[p+1]);
            }
            auto out = linalg::Svd(this->_TNs.back(),true,false);
            this->_TNs.back() = out[1];
            this->S_loc = this->_TNs.size();
        }


        void RegularMPS::S_mvright(){
            if(this->S_loc == this->_TNs.size()){
                return;
            }else if(this->S_loc == -1){
                this->S_loc +=1;
                return;
            }else{
                this->_TNs[this->S_loc].set_rowrank(2);
                auto out = linalg::Svd(this->_TNs[this->S_loc]);
                auto s = out[0];
                this->_TNs[this->S_loc] = out[1];
                auto vt = out[2];
                // boundary:
                if(this->S_loc != this->_TNs.size()-1){
                    this->_TNs[this->S_loc+1] = Contract(Contract(s,vt),this->_TNs[this->S_loc+1]);
                }
                this->S_loc += 1;
            }
        }

        void RegularMPS::S_mvleft(){
            if(this->S_loc == -1){
                return;
            }else if(this->S_loc == this->_TNs.size()){
                this->S_loc -=1;
                return;
            }else{
                this->_TNs[this->S_loc].set_rowrank(1);
                auto out = linalg::Svd(this->_TNs[this->S_loc]);
                auto s = out[0];
                auto u = out[1];
                this->_TNs[this->S_loc] = out[2];

                //boundary:
                if(this->S_loc != 0){
                    this->_TNs[this->S_loc-1] = Contract(this->_TNs[this->S_loc-1],Contract(u,s));
                }
                this->S_loc -= 1;
            }
        }

    }//tn_algo

}// cytnx



