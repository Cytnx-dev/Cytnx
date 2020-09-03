#include <typeinfo>
#include "torcyx.hpp"
#include "ml/CyTensor.hpp"
#include "utils/utils.hpp"
#include "utils/utils_internal_interface.hpp"

using namespace std;

namespace torcyx{
        /*
        // += 
        template<> CyTensor& CyTensor::operator+=<CyTensor>(const CyTensor &rc){
            *this = cytnx::ml::xlinalg::Add(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator+=<cytnx_complex128>(const cytnx_complex128 &rc){
            *this = cytnx::ml::xlinalg::Add(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator+=<cytnx_complex64>(const cytnx_complex64 &rc){
            *this = cytnx::ml::xlinalg::Add(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator+=<cytnx_double>(const cytnx_double &rc){
            *this = cytnx::ml::xlinalg::Add(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator+=<cytnx_float>(const cytnx_float &rc){
            *this = cytnx::ml::xlinalg::Add(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator+=<cytnx_int64>(const cytnx_int64 &rc){
            *this = cytnx::ml::xlinalg::Add(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator+=<cytnx_uint64>(const cytnx_uint64 &rc){
            *this = cytnx::ml::xlinalg::Add(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator+=<cytnx_int32>(const cytnx_int32 &rc){
            *this = cytnx::ml::xlinalg::Add(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator+=<cytnx_uint32>(const cytnx_uint32 &rc){
            *this = cytnx::ml::xlinalg::Add(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator+=<cytnx_int16>(const cytnx_int16 &rc){
            *this = cytnx::ml::xlinalg::Add(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator+=<cytnx_uint16>(const cytnx_uint16 &rc){
            *this = cytnx::ml::xlinalg::Add(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator+=<cytnx_bool>(const cytnx_bool &rc){
            *this = cytnx::ml::xlinalg::Add(*this,rc);
            return *this;
        }

        // -= 
        template<> CyTensor& CyTensor::operator-=<CyTensor>(const CyTensor &rc){
            *this = cytnx::ml::xlinalg::Sub(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator-=<cytnx_complex128>(const cytnx_complex128 &rc){
            *this = cytnx::ml::xlinalg::Sub(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator-=<cytnx_complex64>(const cytnx_complex64 &rc){
            *this = cytnx::ml::xlinalg::Sub(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator-=<cytnx_double>(const cytnx_double &rc){
            *this = cytnx::ml::xlinalg::Sub(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator-=<cytnx_float>(const cytnx_float &rc){
            *this = cytnx::ml::xlinalg::Sub(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator-=<cytnx_int64>(const cytnx_int64 &rc){
            *this = cytnx::ml::xlinalg::Sub(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator-=<cytnx_uint64>(const cytnx_uint64 &rc){
            *this = cytnx::ml::xlinalg::Sub(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator-=<cytnx_int32>(const cytnx_int32 &rc){
            *this = cytnx::ml::xlinalg::Sub(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator-=<cytnx_uint32>(const cytnx_uint32 &rc){
            *this = cytnx::ml::xlinalg::Sub(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator-=<cytnx_int16>(const cytnx_int16 &rc){
            *this = cytnx::ml::xlinalg::Sub(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator-=<cytnx_uint16>(const cytnx_uint16 &rc){
            *this = cytnx::ml::xlinalg::Sub(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator-=<cytnx_bool>(const cytnx_bool &rc){
            *this = cytnx::ml::xlinalg::Sub(*this,rc);
            return *this;
        }

        // *= 
        template<> CyTensor& CyTensor::operator*=<CyTensor>(const CyTensor &rc){
            *this = cytnx::ml::xlinalg::Mul(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator*=<cytnx_complex128>(const cytnx_complex128 &rc){
            *this = cytnx::ml::xlinalg::Mul(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator*=<cytnx_complex64>(const cytnx_complex64 &rc){
            *this = cytnx::ml::xlinalg::Mul(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator*=<cytnx_double>(const cytnx_double &rc){
            *this = cytnx::ml::xlinalg::Mul(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator*=<cytnx_float>(const cytnx_float &rc){
            *this = cytnx::ml::xlinalg::Mul(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator*=<cytnx_int64>(const cytnx_int64 &rc){
            *this = cytnx::ml::xlinalg::Mul(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator*=<cytnx_uint64>(const cytnx_uint64 &rc){
            *this = cytnx::ml::xlinalg::Mul(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator*=<cytnx_int32>(const cytnx_int32 &rc){
            *this = cytnx::ml::xlinalg::Mul(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator*=<cytnx_uint32>(const cytnx_uint32 &rc){
            *this = cytnx::ml::xlinalg::Mul(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator*=<cytnx_int16>(const cytnx_int16 &rc){
            *this = cytnx::ml::xlinalg::Mul(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator*=<cytnx_uint16>(const cytnx_uint16 &rc){
            *this = cytnx::ml::xlinalg::Mul(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator*=<cytnx_bool>(const cytnx_bool &rc){
            *this = cytnx::ml::xlinalg::Mul(*this,rc);
            return *this;
        }

        // /=
        template<> CyTensor& CyTensor::operator/=<CyTensor>(const CyTensor &rc){
            *this = cytnx::ml::xlinalg::Div(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator/=<cytnx_complex128>(const cytnx_complex128 &rc){
            *this = cytnx::ml::xlinalg::Div(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator/=<cytnx_complex64>(const cytnx_complex64 &rc){
            *this = cytnx::ml::xlinalg::Div(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator/=<cytnx_double>(const cytnx_double &rc){
            *this = cytnx::ml::xlinalg::Div(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator/=<cytnx_float>(const cytnx_float &rc){
            *this = cytnx::ml::xlinalg::Div(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator/=<cytnx_int64>(const cytnx_int64 &rc){
            *this = cytnx::ml::xlinalg::Div(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator/=<cytnx_uint64>(const cytnx_uint64 &rc){
            *this = cytnx::ml::xlinalg::Div(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator/=<cytnx_int32>(const cytnx_int32 &rc){
            *this = cytnx::ml::xlinalg::Div(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator/=<cytnx_uint32>(const cytnx_uint32 &rc){
            *this = cytnx::ml::xlinalg::Div(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator/=<cytnx_int16>(const cytnx_int16 &rc){
            *this = cytnx::ml::xlinalg::Div(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator/=<cytnx_uint16>(const cytnx_uint16 &rc){
            *this = cytnx::ml::xlinalg::Div(*this,rc);
            return *this;
        }
        template<> CyTensor& CyTensor::operator/=<cytnx_bool>(const cytnx_bool &rc){
            *this = cytnx::ml::xlinalg::Div(*this,rc);
            return *this;
        }

        CyTensor CyTensor::Pow(const double &p) const{
            return cytnx::ml::xlinalg::Pow(*this,p);
        }
        CyTensor& CyTensor::Pow_(const double &p){
            cytnx::ml::xlinalg::Pow_(*this,p);
            return *this;
        }

        void CyTensor::_Save(std::fstream &f) const{
            cytnx_error_msg(!f.is_open(),"[ERROR][CyTensor] invalid fstream!.%s","\n");
            cytnx_error_msg(this->_impl->cten_type_id==CTenType.Void,"[ERROR][CyTensor] cannot save an uninitialize CyTensor.%s","\n");

            //temporary disable:
            cytnx_error_msg(this->_impl->cten_type_id==CTenType.Sparse,"[ERROR] Save for SparseCyTensor is under developing!!%s","\n");


            unsigned int IDDs = 555;
            f.write((char*)&IDDs,sizeof(unsigned int));
            // first, save common meta data:
            f.write((char*)&this->_impl->cten_type_id,sizeof(int)); // cten type, this is used to determine Sparse/Dense upon load
            f.write((char*)&this->_impl->_is_braket_form,sizeof(bool));
            f.write((char*)&this->_impl->_is_tag,sizeof(bool));
            f.write((char*)&this->_impl->_is_diag,sizeof(bool));
            f.write((char*)&this->_impl->_rowrank,sizeof(cytnx_int64));

            cytnx_uint32 len_name = this->_impl->_name.size();
            f.write((char*)&len_name,sizeof(cytnx_uint32));
            if(len_name!=0){
                const char *cname = this->_impl->_name.c_str();
                f.write(cname,sizeof(char)*len_name);
            }

            cytnx_uint64 rank = this->_impl->_labels.size();
            f.write((char*)&rank,sizeof(cytnx_uint64));
            f.write((char*)&(this->_impl->_labels[0]),sizeof(cytnx_int64)*rank);
            for(cytnx_uint64 i=0;i<rank;i++){
                this->_impl->_bonds[i]._Save(f);
            }

            // second, let dispatch to do remaining saving.
            this->_impl->_save_dispatch(f);

        }
        void CyTensor::_Load(std::fstream &f){
            cytnx_error_msg(!f.is_open(),"[ERROR][CyTensor] invalid fstream%s","\n");
            unsigned int tmpIDDs;
            f.read((char*)&tmpIDDs,sizeof(unsigned int));
            cytnx_error_msg(tmpIDDs!=555,"[ERROR] the object is not a cytnx CyTensor!%s","\n");
            int ctentype;
            f.write((char*)&ctentype,sizeof(int)); // cten type, this is used to determine Sparse/Dense upon load
            if(ctentype==CTenType.Dense){
                this->_impl = boost::intrusive_ptr<CyTensor_base>(new DenseCyTensor());
            }else if(ctentype==CTenType.Sparse){
                //temporary disable:
                cytnx_error_msg(this->_impl->cten_type_id==CTenType.Sparse,"[ERROR] Save for SparseCyTensor is under developing!!%s","\n");
                this->_impl = boost::intrusive_ptr<CyTensor_base>(new SparseCyTensor());
            }else{
                cytnx_error_msg(true,"[ERROR] Unknown CyTensor type!%s","\n");
            }

            f.read((char*)&this->_impl->_is_braket_form,sizeof(bool));
            f.read((char*)&this->_impl->_is_tag,sizeof(bool));
            f.read((char*)&this->_impl->_is_diag,sizeof(bool));
            f.read((char*)&this->_impl->_rowrank,sizeof(cytnx_int64));

            cytnx_uint32 len_name;
            f.read((char*)&len_name,sizeof(cytnx_uint32));
            if(len_name!=0){
                char *cname = (char*)malloc(sizeof(char)*len_name);
                f.read(cname,sizeof(char)*len_name);
                this->_impl->_name = std::string(cname);
                free(cname);
            }

            cytnx_uint64 rank;
            f.read((char*)&rank,sizeof(cytnx_uint64));
            this->_impl->_labels.resize(rank);
            this->_impl->_bonds.resize(rank);

            f.read((char*)&(this->_impl->_labels[0]),sizeof(cytnx_int64)*rank);
            for(cytnx_uint64 i=0;i<rank;i++){
                this->_impl->_bonds[i]._Load(f);
            }


            // second, let dispatch to do remaining loading.
            this->_impl->_load_dispatch(f);



        }

        void CyTensor::Save(const std::string &fname) const{
            fstream f;
            f.open((fname+".cytnx"),ios::out|ios::trunc|ios::binary);
            if(!f.is_open()){
                cytnx_error_msg(true,"[ERROR] invalid file path for save.%s","\n");
            }
            this->_Save(f);
            f.close();

        }
        void CyTensor::Save(const char* fname) const{
            fstream f;
            string ffname = string(fname) + ".cytnx";
            f.open((ffname),ios::out|ios::trunc|ios::binary);
            if(!f.is_open()){
                cytnx_error_msg(true,"[ERROR] invalid file path for save.%s","\n");
            }
            this->_Save(f);
            f.close();
        }

        CyTensor CyTensor::Load(const std::string &fname){
            CyTensor out;
            fstream f;
            f.open(fname,ios::in|ios::binary);
            if(!f.is_open()){
                cytnx_error_msg(true,"[ERROR] invalid file path for load.%s","\n");
            }
            out._Load(f);
            f.close();
            return out;
        }
        CyTensor  CyTensor::Load(const char* fname){
            CyTensor out;
            fstream f;
            f.open(fname,ios::in|ios::binary);
            if(!f.is_open()){
                cytnx_error_msg(true,"[ERROR] invalid file path for load.%s","\n");
            }
            out._Load(f);
            f.close();
            return out;
        }            
        */
    

        std::vector<torch::Tensor> vec_clone(const std::vector<torch::Tensor>& in_vec){
            std::vector<torch::Tensor> out(in_vec.size());
            for(cytnx_uint64 i=0;i<in_vec.size();i++){
                out[i] = in_vec[i].clone();
            }
            return out;
        }

        

}//torcyx


