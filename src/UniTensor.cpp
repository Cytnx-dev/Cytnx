#include <typeinfo>
#include "UniTensor.hpp"
#include "utils/utils.hpp"
#include "utils/utils_internal_interface.hpp"
#include "linalg.hpp"

using namespace std;

namespace cytnx{
    // += 
    template<> UniTensor& UniTensor::operator+=<UniTensor>(const UniTensor &rc){
        *this = cytnx::linalg::Add(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator+=<cytnx_complex128>(const cytnx_complex128 &rc){
        *this = cytnx::linalg::Add(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator+=<cytnx_complex64>(const cytnx_complex64 &rc){
        *this = cytnx::linalg::Add(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator+=<cytnx_double>(const cytnx_double &rc){
        *this = cytnx::linalg::Add(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator+=<cytnx_float>(const cytnx_float &rc){
        *this = cytnx::linalg::Add(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator+=<cytnx_int64>(const cytnx_int64 &rc){
        *this = cytnx::linalg::Add(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator+=<cytnx_uint64>(const cytnx_uint64 &rc){
        *this = cytnx::linalg::Add(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator+=<cytnx_int32>(const cytnx_int32 &rc){
        *this = cytnx::linalg::Add(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator+=<cytnx_uint32>(const cytnx_uint32 &rc){
        *this = cytnx::linalg::Add(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator+=<cytnx_int16>(const cytnx_int16 &rc){
        *this = cytnx::linalg::Add(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator+=<cytnx_uint16>(const cytnx_uint16 &rc){
        *this = cytnx::linalg::Add(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator+=<cytnx_bool>(const cytnx_bool &rc){
        *this = cytnx::linalg::Add(*this,rc);
        return *this;
    }

    // -= 
    template<> UniTensor& UniTensor::operator-=<UniTensor>(const UniTensor &rc){
        *this = cytnx::linalg::Sub(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator-=<cytnx_complex128>(const cytnx_complex128 &rc){
        *this = cytnx::linalg::Sub(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator-=<cytnx_complex64>(const cytnx_complex64 &rc){
        *this = cytnx::linalg::Sub(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator-=<cytnx_double>(const cytnx_double &rc){
        *this = cytnx::linalg::Sub(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator-=<cytnx_float>(const cytnx_float &rc){
        *this = cytnx::linalg::Sub(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator-=<cytnx_int64>(const cytnx_int64 &rc){
        *this = cytnx::linalg::Sub(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator-=<cytnx_uint64>(const cytnx_uint64 &rc){
        *this = cytnx::linalg::Sub(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator-=<cytnx_int32>(const cytnx_int32 &rc){
        *this = cytnx::linalg::Sub(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator-=<cytnx_uint32>(const cytnx_uint32 &rc){
        *this = cytnx::linalg::Sub(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator-=<cytnx_int16>(const cytnx_int16 &rc){
        *this = cytnx::linalg::Sub(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator-=<cytnx_uint16>(const cytnx_uint16 &rc){
        *this = cytnx::linalg::Sub(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator-=<cytnx_bool>(const cytnx_bool &rc){
        *this = cytnx::linalg::Sub(*this,rc);
        return *this;
    }

    // *= 
    template<> UniTensor& UniTensor::operator*=<UniTensor>(const UniTensor &rc){
        *this = cytnx::linalg::Mul(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator*=<cytnx_complex128>(const cytnx_complex128 &rc){
        *this = cytnx::linalg::Mul(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator*=<cytnx_complex64>(const cytnx_complex64 &rc){
        *this = cytnx::linalg::Mul(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator*=<cytnx_double>(const cytnx_double &rc){
        *this = cytnx::linalg::Mul(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator*=<cytnx_float>(const cytnx_float &rc){
        *this = cytnx::linalg::Mul(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator*=<cytnx_int64>(const cytnx_int64 &rc){
        *this = cytnx::linalg::Mul(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator*=<cytnx_uint64>(const cytnx_uint64 &rc){
        *this = cytnx::linalg::Mul(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator*=<cytnx_int32>(const cytnx_int32 &rc){
        *this = cytnx::linalg::Mul(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator*=<cytnx_uint32>(const cytnx_uint32 &rc){
        *this = cytnx::linalg::Mul(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator*=<cytnx_int16>(const cytnx_int16 &rc){
        *this = cytnx::linalg::Mul(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator*=<cytnx_uint16>(const cytnx_uint16 &rc){
        *this = cytnx::linalg::Mul(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator*=<cytnx_bool>(const cytnx_bool &rc){
        *this = cytnx::linalg::Mul(*this,rc);
        return *this;
    }

    // /=
    template<> UniTensor& UniTensor::operator/=<UniTensor>(const UniTensor &rc){
        *this = cytnx::linalg::Div(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator/=<cytnx_complex128>(const cytnx_complex128 &rc){
        *this = cytnx::linalg::Div(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator/=<cytnx_complex64>(const cytnx_complex64 &rc){
        *this = cytnx::linalg::Div(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator/=<cytnx_double>(const cytnx_double &rc){
        *this = cytnx::linalg::Div(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator/=<cytnx_float>(const cytnx_float &rc){
        *this = cytnx::linalg::Div(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator/=<cytnx_int64>(const cytnx_int64 &rc){
        *this = cytnx::linalg::Div(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator/=<cytnx_uint64>(const cytnx_uint64 &rc){
        *this = cytnx::linalg::Div(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator/=<cytnx_int32>(const cytnx_int32 &rc){
        *this = cytnx::linalg::Div(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator/=<cytnx_uint32>(const cytnx_uint32 &rc){
        *this = cytnx::linalg::Div(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator/=<cytnx_int16>(const cytnx_int16 &rc){
        *this = cytnx::linalg::Div(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator/=<cytnx_uint16>(const cytnx_uint16 &rc){
        *this = cytnx::linalg::Div(*this,rc);
        return *this;
    }
    template<> UniTensor& UniTensor::operator/=<cytnx_bool>(const cytnx_bool &rc){
        *this = cytnx::linalg::Div(*this,rc);
        return *this;
    }

    UniTensor UniTensor::Pow(const double &p) const{
        return cytnx::linalg::Pow(*this,p);
    }
    UniTensor& UniTensor::Pow_(const double &p){
        cytnx::linalg::Pow_(*this,p);
        return *this;
    }

    void UniTensor::_Save(std::fstream &f) const{
        cytnx_error_msg(!f.is_open(),"[ERROR][UniTensor] invalid fstream!.%s","\n");
        cytnx_error_msg(this->_impl->uten_type_id==UTenType.Void,"[ERROR][UniTensor] cannot save an uninitialize UniTensor.%s","\n");

        //temporary disable:
        cytnx_error_msg(this->_impl->uten_type_id==UTenType.Sparse,"[ERROR] Save for SparseUniTensor is under developing!!%s","\n");


        unsigned int IDDs = 555;
        f.write((char*)&IDDs,sizeof(unsigned int));
        // first, save common meta data:
        f.write((char*)&this->_impl->uten_type_id,sizeof(int)); // uten type, this is used to determine Sparse/Dense upon load
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
    void UniTensor::_Load(std::fstream &f){
        cytnx_error_msg(!f.is_open(),"[ERROR][UniTensor] invalid fstream%s","\n");
        unsigned int tmpIDDs;
        f.read((char*)&tmpIDDs,sizeof(unsigned int));
        cytnx_error_msg(tmpIDDs!=555,"[ERROR] the object is not a cytnx UniTensor!%s","\n");
        int utentype;
        f.read((char*)&utentype,sizeof(int)); // uten type, this is used to determine Sparse/Dense upon load
        if(utentype==UTenType.Dense){
            this->_impl = boost::intrusive_ptr<UniTensor_base>(new DenseUniTensor());
        }else if(utentype==UTenType.Sparse){
            //temporary disable:
            cytnx_error_msg(this->_impl->uten_type_id==UTenType.Sparse,"[ERROR] Save for SparseUniTensor is under developing!!%s","\n");
            this->_impl = boost::intrusive_ptr<UniTensor_base>(new SparseUniTensor());
        }else{
            cytnx_error_msg(true,"[ERROR] Unknown UniTensor type!%s","\n");
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

    void UniTensor::Save(const std::string &fname) const{
        fstream f;
        f.open((fname+".cytnx"),ios::out|ios::trunc|ios::binary);
        if(!f.is_open()){
            cytnx_error_msg(true,"[ERROR] invalid file path for save.%s","\n");
        }
        this->_Save(f);
        f.close();

    }
    void UniTensor::Save(const char* fname) const{
        fstream f;
        string ffname = string(fname) + ".cytnx";
        f.open((ffname),ios::out|ios::trunc|ios::binary);
        if(!f.is_open()){
            cytnx_error_msg(true,"[ERROR] invalid file path for save.%s","\n");
        }
        this->_Save(f);
        f.close();
    }

    UniTensor UniTensor::Load(const std::string &fname){
        UniTensor out;
        fstream f;
        f.open(fname,ios::in|ios::binary);
        if(!f.is_open()){
            cytnx_error_msg(true,"[ERROR] invalid file path for load.%s","\n");
        }
        out._Load(f);
        f.close();
        return out;
    }
    UniTensor  UniTensor::Load(const char* fname){
        UniTensor out;
        fstream f;
        f.open(fname,ios::in|ios::binary);
        if(!f.is_open()){
            cytnx_error_msg(true,"[ERROR] invalid file path for load.%s","\n");
        }
        out._Load(f);
        f.close();
        return out;
    }            



}
