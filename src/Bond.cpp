#include "Bond.hpp"

using namespace std;
namespace tor10{

    Bond::Bond(tor10_uint64 dim, const std::initializer_list<std::initializer_list<tor10_int64> > &in_qnums,const std::initializer_list<Symmetry> &in_syms){

        tor10_error_msg(dim==0,"%s","[ERROR] Bond cannot have 0 or negative dimension.");
        this->_type = bondType.BD_REGULAR;

        if(in_qnums.size()!=dim){
            tor10_error_msg(true,"%s","[ERROR] invalid qnums. the # of row of qnums list should match the dimension.");
        }

        tor10_int32 Nsym = in_qnums.begin()[0].size();
        std::vector< std::vector<tor10_int64> > in_vec_qnums(dim);
        for(tor10_uint64 i=0;i<dim;i++){
            tor10_error_msg(in_qnums.begin()[i].size() != Nsym,"%s","[ERROR] invalid qnums. the # of column of qnums list should be identify across each row. ");
            in_vec_qnums[i] = in_qnums.begin()[i];
        }

        tor10_error_msg(Nsym==0,"%s","[ERROR] pass empty qnums to initialize Bond is invalid.");
        tor10_error_msg(in_syms.size()!=Nsym,"%s","[ERROR] the number of symmetry should match the column of passed-in qnums.");

        this->_dim = dim;
        this->_qnums = in_vec_qnums;
        this->_syms = in_syms;

        //check qnums match the rule of each symmetry type
        for(tor10_uint64 i=0;i<dim;i++)
            for(tor10_int32 j=0;j<Nsym;j++)
                tor10_error_msg(!this->_syms[j].check_qnum(this->_qnums[i][j]),"[ERROR] invalid qnums @ Symmetry: %d, index %d",j,i);

    }

    Bond::Bond(tor10_uint64 dim, const std::vector<std::vector<tor10_int64> > &in_qnums, const std::vector<Symmetry> &in_syms){
        tor10_error_msg(dim==0,"%s","[ERROR] Bond cannot have 0 or negative dimension.");
        this->_type = bondType::BD_REGULAR;

        if(in_qnums.size()!=dim){
            tor10_error_msg(true,"%s","[ERROR] invalid qnums. the # of row of qnums list should match the dimension.");
        }

        tor10_int32 Nsym = in_qnums[0].size();
        for(tor10_uint64 i=0;i<dim;i++){
            tor10_error_msg(in_qnums[i].size() != Nsym,"%s","[ERROR] invalid qnums. the # of column of qnums list should be identify across each row. ");
        }

        tor10_error_msg(Nsym==0,"%s","[ERROR] pass empty qnums to initialize Bond is invalid.");
        tor10_error_msg(in_syms.size()!=Nsym,"%s","[ERROR] the number of symmetry should match the column of passed-in qnums.");
        this->_dim = dim;
        this->_qnums = in_qnums;
        this->_syms = in_syms;

        //check qnums match the rule of each symmetry type
        for(tor10_uint64 i=0;i<dim;i++)
            for(tor10_int32 j=0;j<Nsym;j++)
                tor10_error_msg(!this->_syms[j].check_qnum(this->_qnums[i][j]),"[ERROR] invalid qnums @ Symmetry: %d, index %d",j,i);
                

    }


    std::ostream& operator<<(std::ostream &os,const Bond &bin){
        os << "Dim = " << bin.dim() << " |";
        if(bin.get_type()==bondType::BD_REGULAR){
            os << "type: REGULAR " << std::endl;
        }else if(bin.get_type()==bondType::BD_BRA){
            os << "type: <BRA     " << std::endl;
        }else if(bin.get_type()==bondType::BD_KET){
            os << "type: KET>     " << std::endl;
        }else{
            tor10_error_msg(1,"%s","[ERROR] internal error.");
        }
        //os << bin.get_syms().size() << endl;

        for(tor10_int32 i=0;i<bin.Nsym();i++){
            os << " " << bin.get_syms()[i].stype() << ":: ";
            for(tor10_int32 j=0;j<bin.dim();j++){
                printf(" %+2d",bin.get_qnums()[j][i]);
            }
            os << std::endl;
        }
        return os;
    }

}
