#include "linalg/linalg_internal_cpu/Arithmic_internal_cpu.hpp"

namespace tor10{
    namespace linalg_internal{
        

        void Arithmic_internal_cpu_cdtcd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_cdtcd(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_cdtcd(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_cdtcd(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_cdtcd(out,Lin,Rin,len);
        }
        void Arithmic_internal_cpu_cdtcf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_cdtcf(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_cdtcf(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_cdtcf(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_cdtcf(out,Lin,Rin,len);
        }
        void Arithmic_internal_cpu_cdtd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_cdtd(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_cdtd(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_cdtd(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_cdtd(out,Lin,Rin,len);
        }
        void Arithmic_internal_cpu_cdtf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_cdtf(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_cdtf(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_cdtf(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_cdtf(out,Lin,Rin,len);
        }
        void Arithmic_internal_cpu_cdtu64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_cdtu64(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_cdtu64(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_cdtu64(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_cdtu64(out,Lin,Rin,len);
        }
        void Arithmic_internal_cpu_cdti64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_cdti64(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_cdti64(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_cdti64(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_cdti64(out,Lin,Rin,len);
        }
        void Arithmic_internal_cpu_cdtu32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_cdtu32(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_cdtu32(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_cdtu32(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_cdtu32(out,Lin,Rin,len);
        }
        void Arithmic_internal_cpu_cdti32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_cdti32(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_cdti32(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_cdti32(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_cdti32(out,Lin,Rin,len);
        }

        void Arithmic_internal_cpu_cftcd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_cftcd(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_cftcd(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_cftcd(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_cftcd(out,Lin,Rin,len);
        }
        void Arithmic_internal_cpu_cftcf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_cftcf(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_cftcf(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_cftcf(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_cftcf(out,Lin,Rin,len);
        }
        void Arithmic_internal_cpu_cftd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_cftd(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_cftd(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_cftd(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_cftd(out,Lin,Rin,len);
        }
        void Arithmic_internal_cpu_cftf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_cftf(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_cftf(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_cftf(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_cftf(out,Lin,Rin,len);
        }
        void Arithmic_internal_cpu_cftu64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_cftu64(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_cftu64(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_cftu64(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_cftu64(out,Lin,Rin,len);
        }
        void Arithmic_internal_cpu_cfti64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_cfti64(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_cfti64(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_cfti64(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_cfti64(out,Lin,Rin,len);
        }
        void Arithmic_internal_cpu_cftu32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_cftu32(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_cftu32(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_cftu32(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_cftu32(out,Lin,Rin,len);
        }
        void Arithmic_internal_cpu_cfti32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_cfti32(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_cfti32(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_cfti32(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_cfti32(out,Lin,Rin,len);
        }

        void Arithmic_internal_cpu_dtcd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_dtcd(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_dtcd(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_dtcd(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_dtcd(out,Lin,Rin,len);
        }
        void Arithmic_internal_cpu_dtcf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_dtcf(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_dtcf(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_dtcf(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_dtcf(out,Lin,Rin,len);
        }
        void Arithmic_internal_cpu_dtd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_dtd(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_dtd(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_dtd(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_dtd(out,Lin,Rin,len);
        }
        void Arithmic_internal_cpu_dtf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_dtf(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_dtf(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_dtf(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_dtf(out,Lin,Rin,len);
        }
        void Arithmic_internal_cpu_dtu64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_dtu64(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_dtu64(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_dtu64(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_dtu64(out,Lin,Rin,len);
        }
        void Arithmic_internal_cpu_dti64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_dti64(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_dti64(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_dti64(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_dti64(out,Lin,Rin,len);
        }
        void Arithmic_internal_cpu_dtu32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_dtu32(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_dtu32(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_dtu32(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_dtu32(out,Lin,Rin,len);
        }
        void Arithmic_internal_cpu_dti32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_dti32(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_dti32(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_dti32(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_dti32(out,Lin,Rin,len);
        }

        void Arithmic_internal_cpu_ftcd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_ftcd(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_ftcd(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_ftcd(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_ftcd(out,Lin,Rin,len);
        }
        void Arithmic_internal_cpu_ftcf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_ftcf(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_ftcf(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_ftcf(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_ftcf(out,Lin,Rin,len);
	}
        void Arithmic_internal_cpu_ftd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_ftd(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_ftd(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_ftd(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_ftd(out,Lin,Rin,len);
	}
        void Arithmic_internal_cpu_ftf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_ftf(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_ftf(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_ftf(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_ftf(out,Lin,Rin,len);
}
        void Arithmic_internal_cpu_ftu64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_ftu64(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_ftu64(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_ftu64(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_ftu64(out,Lin,Rin,len);

	}
        void Arithmic_internal_cpu_fti64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_fti64(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_fti64(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_fti64(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_fti64(out,Lin,Rin,len);

}
        void Arithmic_internal_cpu_ftu32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){

            if(type==0)      tor10::linalg_internal::Add_internal_cpu_ftu32(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_ftu32(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_ftu32(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_ftu32(out,Lin,Rin,len);

}
        void Arithmic_internal_cpu_fti32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_fti32(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_fti32(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_fti32(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_fti32(out,Lin,Rin,len);

}

        void Arithmic_internal_cpu_u64tcd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){

            if(type==0)      tor10::linalg_internal::Add_internal_cpu_u64tcd(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_u64tcd(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_u64tcd(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_u64tcd(out,Lin,Rin,len);

}
        void Arithmic_internal_cpu_u64tcf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_u64tcf(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_u64tcf(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_u64tcf(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_u64tcf(out,Lin,Rin,len);


}
        void Arithmic_internal_cpu_u64td(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_u64td(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_u64td(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_u64td(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_u64td(out,Lin,Rin,len);


}
        void Arithmic_internal_cpu_u64tf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_u64tf(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_u64tf(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_u64tf(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_u64tf(out,Lin,Rin,len);


}
        void Arithmic_internal_cpu_u64tu64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_u64tu64(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_u64tu64(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_u64tu64(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_u64tu64(out,Lin,Rin,len);


}
        void Arithmic_internal_cpu_u64ti64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_u64ti64(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_u64ti64(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_u64ti64(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_u64ti64(out,Lin,Rin,len);


}
        void Arithmic_internal_cpu_u64tu32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_u64tu32(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_u64tu32(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_u64tu32(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_u64tu32(out,Lin,Rin,len);


}
        void Arithmic_internal_cpu_u64ti32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_u64ti32(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_u64ti32(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_u64ti32(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_u64ti32(out,Lin,Rin,len);


}

        void Arithmic_internal_cpu_i64tcd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_i64tcd(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_i64tcd(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_i64tcd(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_i64tcd(out,Lin,Rin,len);


}
        void Arithmic_internal_cpu_i64tcf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_i64tcf(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_i64tcf(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_i64tcf(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_i64tcf(out,Lin,Rin,len);


}
        void Arithmic_internal_cpu_i64td(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_i64td(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_i64td(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_i64td(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_i64td(out,Lin,Rin,len);


}
        void Arithmic_internal_cpu_i64tf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_i64tf(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_i64tf(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_i64tf(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_i64tf(out,Lin,Rin,len);


}
        void Arithmic_internal_cpu_i64tu64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_i64tu64(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_i64tu64(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_i64tu64(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_i64tu64(out,Lin,Rin,len);


}
        void Arithmic_internal_cpu_i64ti64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_i64ti64(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_i64ti64(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_i64ti64(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_i64ti64(out,Lin,Rin,len);


}
        void Arithmic_internal_cpu_i64tu32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_i64tu32(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_i64tu32(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_i64tu32(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_i64tu32(out,Lin,Rin,len);


}
        void Arithmic_internal_cpu_i64ti32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_i64ti32(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_i64ti32(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_i64ti32(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_i64ti32(out,Lin,Rin,len);


}

        void Arithmic_internal_cpu_u32tcd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_u32tcd(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_u32tcd(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_u32tcd(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_u32tcd(out,Lin,Rin,len);


}
        void Arithmic_internal_cpu_u32tcf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_u32tcf(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_u32tcf(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_u32tcf(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_u32tcf(out,Lin,Rin,len);


}
        void Arithmic_internal_cpu_u32td(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_u32td(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_u32td(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_u32td(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_u32td(out,Lin,Rin,len);


}
        void Arithmic_internal_cpu_u32tf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_u32tf(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_u32tf(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_u32tf(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_u32tf(out,Lin,Rin,len);


}
        void Arithmic_internal_cpu_u32tu64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_u32tu64(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_u32tu64(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_u32tu64(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_u32tu64(out,Lin,Rin,len);


}
        void Arithmic_internal_cpu_u32ti64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_u32ti64(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_u32ti64(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_u32ti64(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_u32ti64(out,Lin,Rin,len);


}
        void Arithmic_internal_cpu_u32tu32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_u32tu32(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_u32tu32(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_u32tu32(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_u32tu32(out,Lin,Rin,len);


}
        void Arithmic_internal_cpu_u32ti32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_u32ti32(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_u32ti32(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_u32ti32(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_u32ti32(out,Lin,Rin,len);


}

        void Arithmic_internal_cpu_i32tcd(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_i32tcd(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_i32tcd(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_i32tcd(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_i32tcd(out,Lin,Rin,len);


}
        void Arithmic_internal_cpu_i32tcf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_i32tcf(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_i32tcf(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_i32tcf(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_i32tcf(out,Lin,Rin,len);


}
        void Arithmic_internal_cpu_i32td(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_i32td(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_i32td(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_i32td(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_i32td(out,Lin,Rin,len);


}
        void Arithmic_internal_cpu_i32tf(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_i32tf(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_i32tf(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_i32tf(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_i32tf(out,Lin,Rin,len);


}
        void Arithmic_internal_cpu_i32tu64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_i32tu64(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_i32tu64(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_i32tu64(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_i32tu64(out,Lin,Rin,len);


}
        void Arithmic_internal_cpu_i32ti64(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_i32ti64(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_i32ti64(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_i32ti64(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_i32ti64(out,Lin,Rin,len);


}
        void Arithmic_internal_cpu_i32tu32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_i32tu32(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_i32tu32(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_i32tu32(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_i32tu32(out,Lin,Rin,len);


}
        void Arithmic_internal_cpu_i32ti32(boost::intrusive_ptr<Storage_base> & out, boost::intrusive_ptr<Storage_base> & Lin, boost::intrusive_ptr<Storage_base> & Rin, const unsigned long long &len, const char &type){
            if(type==0)      tor10::linalg_internal::Add_internal_cpu_i32ti32(out,Lin,Rin,len);
            else if(type==1) tor10::linalg_internal::Mul_internal_cpu_i32ti32(out,Lin,Rin,len);
            else if(type==2) tor10::linalg_internal::Sub_internal_cpu_i32ti32(out,Lin,Rin,len);
            else             tor10::linalg_internal::Div_internal_cpu_i32ti32(out,Lin,Rin,len);


}


   }
}


