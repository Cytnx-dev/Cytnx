#include "Bond.hpp"

using namespace std;
namespace cytnx{

    void Bond_impl::Init(const cytnx_uint64 &dim, const bondType &bd_type, const std::vector<std::vector<cytnx_int64> > &in_qnums,const std::vector<Symmetry> &in_syms){
       
        cytnx_error_msg(dim==0,"%s","[ERROR] Bond_impl cannot have 0 or negative dimension.");
        //check is symmetry:
        if(in_qnums.size()==0){
            cytnx_error_msg(in_syms.size()!=0,"[ERROR] No qnums assigned, but with symmetry provided.%s","\n");
            this->_type = bd_type;
            this->_dim = dim;

        }else{
                

            cytnx_uint64 Ndim = in_qnums.begin()[0].size();
            for(cytnx_uint64 i=0;i<in_qnums.size();i++){
                cytnx_error_msg(in_qnums.begin()[i].size() != Ndim,"%s","[ERROR] invalid qnums. the # of column of qnums list should be identify across each row. ");
            }
            
            //cytnx_error_msg(Nsym==0,"%s","[ERROR] pass empty qnums to initialize Bond_impl is invalid.");
            if(in_syms.size()==0){
                this->_syms.clear();
                for(cytnx_uint64 i=0;i<in_qnums.size();i++){
                    this->_syms.push_back(Symmetry::U1());
                }
                
            }else{
                cytnx_error_msg(in_syms.size()!=in_qnums.size(),"%s","[ERROR] the number of symmetry should match the # of rows of passed-in qnums.");
                this->_syms = vec_clone(in_syms);
            }
            this->_dim = dim;
            this->_qnums = in_qnums;
            this->_type = bd_type;

            //check qnums match the rule of each symmetry type
            for(cytnx_uint64 i=0;i<in_qnums.size();i++)
                cytnx_error_msg(!this->_syms[i].check_qnums(this->_qnums[i]),"[ERROR] invalid qnums @ Symmetry: %d\n",i);
        }
    }

    /*
    void Bond_impl::Init(const cytnx_uint64 &dim, const std::initializer_list<std::initializer_list<cytnx_int64> > &in_qnums,const std::initializer_list<Symmetry> &in_syms,const bondType &bd_type){


        std::vector< std::vector<cytnx_int64> > in_vec_qnums(in_qnums.size());
        for(cytnx_uint64 i=0;i<in_qnums.size();i++){
            //cytnx_error_msg(in_qnums.begin()[i].size() != Nsym,"%s","[ERROR] invalid qnums. the # of column of qnums list should be identify across each row. ");
            in_vec_qnums[i] = in_qnums.begin()[i];
        }

        std::vector<Symmetry> in_vec_syms = in_syms;

        this->Init(dim,in_vec_qnums,in_vec_syms,bd_type);

    }
    */


    bool Bond::operator==(const Bond &rhs){
        if(this->dim() != rhs.dim()) return false;
        if(this->type() != rhs.type()) return false;
        if(this->Nsym() != rhs.Nsym()) return false;
        if(this->Nsym()!=0){
            if(this->syms() != rhs.syms()) return false;
            if(this->qnums().size() != rhs.qnums().size()) return false;
            for(cytnx_uint64 i=0;i<this->qnums().size();i++){
                if(this->qnums()[i] != rhs.qnums()[i]) return false;
            }
        }
        return true;        
    }

    std::ostream& operator<<(std::ostream &os,const Bond &bin){
        os << "Dim = " << bin.dim() << " |";
        if(bin.type()==bondType::BD_REG){
            os << "type: REGULAR " << std::endl;
        }else if(bin.type()==bondType::BD_BRA){
            os << "type: <BRA     " << std::endl;
        }else if(bin.type()==bondType::BD_KET){
            os << "type: KET>     " << std::endl;
        }else{
            cytnx_error_msg(1,"%s","[ERROR] internal error.");
        }
        //os << bin.get_syms().size() << endl;

        for(cytnx_int32 i=0;i<bin.Nsym();i++){
            os << " " << bin.syms()[i].stype_str() << ":: ";
            for(cytnx_int32 j=0;j<bin.dim();j++){
                printf(" %+2d",bin.qnums()[i][j]);
            }
            os << std::endl;
        }
        return os;
    }

}
