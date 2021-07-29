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
            os << "physBD dim :\n";

            // print Sloc indicator:
            if(this->S_loc==-1){
                os << ".[";
            }else{
                os << " [";
            }
            for(int i=0;i<this->_TNs.size();i++){
                os << " ";  
                if(this->S_loc==i) os << "'" << this->_TNs[i].shape()[1] << "'";
                else  os << this->_TNs[i].shape()[1];
            }
            if(this->S_loc==this->_TNs.size()){
                os << " ].\n";
            }else{
                os << "] \n";
            }

            os << "virtBD dim : " << this->virt_dim << endl;
            os << endl;
            return os;
        }


        void RegularMPS::Init(const cytnx_uint64 &N, const std::vector<cytnx_uint64> &vphys_dim, const cytnx_uint64 &virt_dim, const cytnx_int64 &dtype){
            //checking:
            cytnx_error_msg(N==0,"[ERROR][RegularMPS] number of site N cannot be ZERO.%s","\n");
            cytnx_error_msg(N!=vphys_dim.size(),"[ERROR] RegularMPS vphys_dim.size() should be equal to N.%s","\n"); 
            cytnx_error_msg(dtype!=Type.Double,"[ERROR][RegularMPS] currently only Double dtype is support.%s","\n");

            this->virt_dim = virt_dim;            

            const cytnx_uint64& chi = virt_dim;
            
            this->_TNs.resize(N);
            this->_TNs[0] = UniTensor(cytnx::random::normal({1, vphys_dim[0], min(chi, vphys_dim[0])}, 0., 1.),2);
            cytnx_uint64 dim1,dim2,dim3;

            cytnx_uint64 DR = 1;
            cytnx_int64 k_ov=0;
            for(cytnx_int64 k=N-1;k>0;k--){
                if(std::numeric_limits<cytnx_uint64>::max()/vphys_dim[k] >= DR){
                    k_ov = k;
                    break;
                }else{
                    DR*= vphys_dim[k];
                }
            }


            for(cytnx_int64 k=1; k<N; k++){
                dim1 = this->_TNs[k-1].shape()[2]; dim2 = vphys_dim[k];
                if(k<=k_ov){
                    dim3 = std::min(chi, cytnx_uint64(dim1 * dim2));
                }else{
                    DR/=vphys_dim[k];
                    dim3 = std::min(std::min(chi, cytnx_uint64(dim1 * dim2)),DR);
                }
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


        void RegularMPS::_save_dispatch(fstream &f){

            cytnx_uint64 N = this->_TNs.size();
            f.write((char*)&N,sizeof(cytnx_uint64));

            // save UniTensor one by one:
            for(cytnx_uint64 i=0;i<N;i++){
                this->_TNs[i]._Save(f);
            }            

        }
        void RegularMPS::_load_dispatch(fstream &f){
            cytnx_uint64 N;

            
            f.read((char*)&N,sizeof(cytnx_uint64));
            this->_TNs.resize(N);
                
            // Load UniTensor one by one:
            for(cytnx_uint64 i=0;i<N;i++){
                this->_TNs[i]._Load(f);
            }


        }


    }//tn_algo

}// cytnx



