#include <typeinfo>
#include "Tensor.hpp"
#include "utils/utils_internal_interface.hpp"
#include "linalg.hpp"
#include "utils/is.hpp"
using namespace std;

namespace cytnx{        

   
    //----------------------------------------------
    //Tproxy
    
    Tensor Tensor::Tproxy::operator+=(const Tensor::Tproxy &rc){
        Tensor self;
        self._impl = _insimpl->get(_accs);
        //self += Tensor(rc);
        self._impl->storage() = cytnx::linalg::Add(self,Tensor(rc))._impl->storage();

        _insimpl->set(_accs,self._impl);
        self._impl = this->_insimpl;
        return self;
    }
    Tensor Tensor::Tproxy::operator-=(const Tensor::Tproxy &rc){
        Tensor self;
        self._impl = _insimpl->get(_accs);
        //self += Tensor(rc);
        self._impl->storage() = cytnx::linalg::Sub(self,Tensor(rc))._impl->storage();

        _insimpl->set(_accs,self._impl);
        self._impl = this->_insimpl;
        return self;
    }
    Tensor Tensor::Tproxy::operator/=(const Tensor::Tproxy &rc){
        Tensor self;
        self._impl = _insimpl->get(_accs);
        //self += Tensor(rc);
        self._impl->storage() = cytnx::linalg::Div(self,Tensor(rc))._impl->storage();

        _insimpl->set(_accs,self._impl);
        self._impl = this->_insimpl;
        return self;
    }
    Tensor Tensor::Tproxy::operator*=(const Tensor::Tproxy &rc){
        Tensor self;
        self._impl = _insimpl->get(_accs);
        //self += Tensor(rc);
        self._impl->storage() = cytnx::linalg::Mul(self,Tensor(rc))._impl->storage();

        _insimpl->set(_accs,self._impl);
        self._impl = this->_insimpl;
        return self;
    }

    //ADD
    Tensor Tensor::Tproxy::operator+(const cytnx_complex128 &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Add(rc);
    }   
    Tensor Tensor::Tproxy::operator+(const cytnx_complex64 &rc)  const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Add(rc);
    }   
    Tensor Tensor::Tproxy::operator+(const cytnx_double &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Add(rc);
    }   
    Tensor Tensor::Tproxy::operator+(const cytnx_float &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Add(rc);
    }   
    Tensor Tensor::Tproxy::operator+(const cytnx_uint64 &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Add(rc);
    }   
    Tensor Tensor::Tproxy::operator+(const cytnx_int64 &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Add(rc);
    } 
    Tensor Tensor::Tproxy::operator+(const cytnx_uint32 &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Add(rc);
    } 
    Tensor Tensor::Tproxy::operator+(const cytnx_int32 &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Add(rc);
    } 
    Tensor Tensor::Tproxy::operator+(const cytnx_uint16 &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Add(rc);
    } 
    Tensor Tensor::Tproxy::operator+(const cytnx_int16 &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Add(rc);
    } 
    Tensor Tensor::Tproxy::operator+(const cytnx_bool &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Add(rc);
    } 
   
    Tensor Tensor::Tproxy::operator+(const Tproxy &rc) const{
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return cytnx::linalg::Add(out,Tensor(rc));
    }

    //SUB:
    Tensor Tensor::Tproxy::operator-(const cytnx_complex128 &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Sub(rc);
    }   
    Tensor Tensor::Tproxy::operator-(const cytnx_complex64 &rc)  const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Sub(rc);
    }   
    Tensor Tensor::Tproxy::operator-(const cytnx_double &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Sub(rc);
    }   
    Tensor Tensor::Tproxy::operator-(const cytnx_float &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Sub(rc);
    }   
    Tensor Tensor::Tproxy::operator-(const cytnx_uint64 &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Sub(rc);
    }   
    Tensor Tensor::Tproxy::operator-(const cytnx_int64 &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Sub(rc);
    } 
    Tensor Tensor::Tproxy::operator-(const cytnx_uint32 &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Sub(rc);
    } 
    Tensor Tensor::Tproxy::operator-(const cytnx_int32 &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Sub(rc);
    } 
    Tensor Tensor::Tproxy::operator-(const cytnx_uint16 &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Sub(rc);
    } 
    Tensor Tensor::Tproxy::operator-(const cytnx_int16 &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Sub(rc);
    } 
    Tensor Tensor::Tproxy::operator-(const cytnx_bool &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Sub(rc);
    } 
    Tensor Tensor::Tproxy::operator-(const Tproxy &rc) const{
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return cytnx::linalg::Sub(out,Tensor(rc));
    }
    Tensor Tensor::Tproxy::operator-() const{
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Mul(-1);
    }


    // MUL
    Tensor Tensor::Tproxy::operator*(const cytnx_complex128 &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Mul(rc);
    }   
    Tensor Tensor::Tproxy::operator*(const cytnx_complex64 &rc)  const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Mul(rc);
    }   
    Tensor Tensor::Tproxy::operator*(const cytnx_double &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Mul(rc);
    }   
    Tensor Tensor::Tproxy::operator*(const cytnx_float &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Mul(rc);
    }   
    Tensor Tensor::Tproxy::operator*(const cytnx_uint64 &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Mul(rc);
    }   
    Tensor Tensor::Tproxy::operator*(const cytnx_int64 &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Mul(rc);
    } 
    Tensor Tensor::Tproxy::operator*(const cytnx_uint32 &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Mul(rc);
    } 
    Tensor Tensor::Tproxy::operator*(const cytnx_int32 &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Mul(rc);
    } 
    Tensor Tensor::Tproxy::operator*(const cytnx_uint16 &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Mul(rc);
    } 
    Tensor Tensor::Tproxy::operator*(const cytnx_int16 &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Mul(rc);
    } 
    Tensor Tensor::Tproxy::operator*(const cytnx_bool &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Mul(rc);
    } 
    Tensor Tensor::Tproxy::operator*(const Tproxy &rc) const{
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return cytnx::linalg::Mul(out,Tensor(rc));
    }

    //DIV
    Tensor Tensor::Tproxy::operator/(const cytnx_complex128 &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Div(rc);
    }   
    Tensor Tensor::Tproxy::operator/(const cytnx_complex64 &rc)  const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Div(rc);
    }   
    Tensor Tensor::Tproxy::operator/(const cytnx_double &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Div(rc);
    }   
    Tensor Tensor::Tproxy::operator/(const cytnx_float &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Div(rc);
    }   
    Tensor Tensor::Tproxy::operator/(const cytnx_uint64 &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Div(rc);
    }   
    Tensor Tensor::Tproxy::operator/(const cytnx_int64 &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Div(rc);
    } 
    Tensor Tensor::Tproxy::operator/(const cytnx_uint32 &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Div(rc);
    } 
    Tensor Tensor::Tproxy::operator/(const cytnx_int32 &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Div(rc);
    } 
    Tensor Tensor::Tproxy::operator/(const cytnx_uint16 &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Div(rc);
    } 
    Tensor Tensor::Tproxy::operator/(const cytnx_int16 &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Div(rc);
    } 
    Tensor Tensor::Tproxy::operator/(const cytnx_bool &rc) const{//{return this->_operatorADD(rc);};
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.Div(rc);
    } 
    Tensor Tensor::Tproxy::operator/(const Tproxy &rc) const{
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return cytnx::linalg::Div(out,Tensor(rc));
    }


    //-----------------------------------------------
    void Tensor_impl::Init(const std::vector<cytnx_uint64> &shape, const unsigned int &dtype, int device){
        //check:
        cytnx_error_msg(dtype>=N_Type,"%s","[ERROR] invalid argument: dtype");
        cytnx_error_msg(shape.size()==0,"%s","[ERROR] invalid argument: shape. Must at least have one element.");
        cytnx_uint64 Nelem= 1;
        for(int i=0;i<shape.size();i++){
            cytnx_error_msg(shape[i]==0,"%s","[ERROR] shape cannot have 0 dimension in any rank.");
            Nelem *= shape[i]; 
        }
        //this->_storage = __SII.USIInit[dtype]();
        this->_storage.Init(Nelem,dtype,device);
        this->_shape = shape;
        this->_mapper = vec_range(shape.size());
        this->_invmapper = this->_mapper;
        this->_contiguous = true;

    }
    void Tensor_impl::Init(const Storage &in){
        cytnx_error_msg(in.dtype()==Type.Void,"[ERROR] cannot init Tensor using un-initialized Storage%s","\n");
        this->_storage = in;
        this->_shape.clear(); this->_shape.push_back(in.size());
        this->_mapper.clear(); this->_mapper.push_back(0);
        this->_invmapper = this->_mapper;
        this->_contiguous=true;
    }    


    boost::intrusive_ptr<Tensor_impl> Tensor_impl::permute(const std::vector<cytnx_uint64> &rnks){

        

        //check::
        if(rnks.size()!=this->_shape.size()){
            cytnx_error_msg(true,"%s","reshape a tensor with a specify shape that does not match with the shape of the incident tensor.");
        }

        if(vec_unique(rnks).size()!=rnks.size()){
            cytnx_error_msg(true,"%s","tensor permute with duplicated index.\n");
        }

        std::vector<cytnx_uint64> new_fwdmap(this->_shape.size());
        std::vector<cytnx_uint64> new_shape(this->_shape.size());
        std::vector<cytnx_uint64> new_idxmap(this->_shape.size());

        //for(int i=0;i<this->_shape.size();i++)
        //    std::cout << this->_mapper[i] << " " << this->_invmapper[i] << std::endl;                


        for(cytnx_uint32 i=0;i<rnks.size();i++){
            if(rnks[i] >= rnks.size()){
                cytnx_error_msg(1,"%s","reshape a tensor with invalid rank index.");
            }
            //std::cout << this->_mapper[rnks[i]] << " " << i << std::endl;
            new_idxmap[this->_mapper[rnks[i]]] = i;
            new_fwdmap[i] = this->_mapper[rnks[i]];
            new_shape[i] = this->_shape[rnks[i]];

        }

        boost::intrusive_ptr<Tensor_impl> out(new Tensor_impl());
        out->_invmapper = new_idxmap;
        out->_shape = new_shape;
        out->_mapper = new_fwdmap;

        ///checking if permute back to contiguous:
        bool iconti=true;
        for(cytnx_uint32 i=0;i<rnks.size();i++){
            if(new_fwdmap[i]!=new_idxmap[i]){iconti = false; break;}
            if(new_fwdmap[i] != i){iconti=false; break;}
        }
        out->_contiguous= iconti;
        
        //ref storage
        out->_storage = this->_storage;
        return out;
    }

    void Tensor_impl::permute_(const std::vector<cytnx_uint64> &rnks){
        //check::
        if(rnks.size()!=this->_shape.size()){
            cytnx_error_msg(true,"%s","reshape a tensor with a specify shape that does not match with the shape of the incident tensor.");
        }

        if(vec_unique(rnks).size()!=rnks.size()){
            cytnx_error_msg(true,"%s","tensor permute with duplicated index.\n");
        }

        std::vector<cytnx_uint64> new_fwdmap(this->_shape.size());
        std::vector<cytnx_uint64> new_shape(this->_shape.size());
        std::vector<cytnx_uint64> new_idxmap(this->_shape.size());

        //for(int i=0;i<this->_shape.size();i++)
        //    std::cout << this->_mapper[i] << " " << this->_invmapper[i] << std::endl;                


        for(cytnx_uint32 i=0;i<rnks.size();i++){
            if(rnks[i] >= rnks.size()){
                cytnx_error_msg(1,"%s","reshape a tensor with invalid rank index.");
            }
            //std::cout << this->_mapper[rnks[i]] << " " << i << std::endl;
            new_idxmap[this->_mapper[rnks[i]]] = i;
            new_fwdmap[i] = this->_mapper[rnks[i]];
            new_shape[i] = this->_shape[rnks[i]];

        }

        this->_invmapper = new_idxmap;
        this->_shape = new_shape;
        this->_mapper = new_fwdmap;

        ///checking if permute back to contiguous:
        bool iconti=true;
        for(cytnx_uint32 i=0;i<rnks.size();i++){
            if(new_fwdmap[i]!=new_idxmap[i]){iconti = false; break;}
            if(new_fwdmap[i] != i){iconti=false; break;}
        }
        this->_contiguous= iconti;
    }            

    

    boost::intrusive_ptr<Tensor_impl> Tensor_impl::get(const std::vector<cytnx::Accessor> &accessors){
        cytnx_error_msg(accessors.size() > this->_shape.size(), "%s", "The input indexes rank is out of range! (>Tensor's rank).");

        std::vector<cytnx::Accessor> acc = accessors;
        for(int i=0;i<this->_shape.size()-accessors.size();i++){
            acc.push_back(Accessor::all());    
        }

        vector<cytnx_uint64> get_shape(acc.size());

        //vector<cytnx_uint64> new_shape;
        std::vector<std::vector<cytnx_uint64> > locators(acc.size());
        for(cytnx_uint32 i=0;i<acc.size();i++){
            acc[i].get_len_pos(this->_shape[i],get_shape[i],locators[i]); 
            //std::cout << this->_shape[i] << " " << get_shape[i] << "|";
            //for(int j=0;j<locators[i].size();j++) std::cout << locators[i][j] << " ";
            //std::cout << std::endl;
        }   



        boost::intrusive_ptr<Tensor_impl> out( new Tensor_impl());
        out->Init(get_shape,this->dtype(),this->device());
        
        this->storage()._impl->GetElem_byShape(out->storage()._impl,this->shape(),this->_mapper,get_shape,locators);

        vector<cytnx_int64> new_shape;
        for(cytnx_uint32 i=0;i<acc.size();i++)
            if(get_shape[i]!=1) new_shape.push_back(get_shape[i]);

        if(new_shape.size()==0) out->reshape_({1});
        else out->reshape_(new_shape);
        return out;

    }
    
    void Tensor_impl::set(const std::vector<cytnx::Accessor> &accessors, const boost::intrusive_ptr<Tensor_impl> &rhs){
        cytnx_error_msg(accessors.size() > this->_shape.size(), "%s", "The input indexes rank is out of range! (>Tensor's rank).");

        vector<cytnx::Accessor> acc = accessors;
        for(int i=0;i<this->_shape.size()-accessors.size();i++){
            acc.push_back(Accessor::all());    
        }

        vector<cytnx_uint64> get_shape(acc.size());
        //vector<cytnx_uint64> new_shape;

        std::vector<std::vector<cytnx_uint64> > locators(acc.size());
        for(cytnx_uint32 i=0;i<acc.size();i++){
            acc[i].get_len_pos(this->_shape[i],get_shape[i],locators[i]); 
            //std::cout << this->_shape[i] << " " << get_shape[i] << "|";
            //for(int j=0;j<locators[i].size();j++) std::cout << locators[i][j] << " ";
            //std::cout << std::endl;
        }   

        //remove single dim
        vector<cytnx_uint64> new_shape;
        for(cytnx_uint32 i=0;i<acc.size();i++)
            if(get_shape[i]!=1) new_shape.push_back(get_shape[i]);

        if(new_shape.size()==0) new_shape.push_back(1);
        
        // check size:
        cytnx_error_msg(new_shape != rhs->shape(), "[ERROR][Tensor.set_elems]%s","inconsistent shape");


        //boost::intrusive_ptr<Tensor_impl> out( new Tensor_impl());
        //out->Init(get_shape,this->dtype(),this->device());

        this->storage()._impl->SetElem_byShape(rhs->storage()._impl,this->shape(),this->_mapper,get_shape,locators,false);
    }

    template<class T>
    void Tensor_impl::set(const std::vector<cytnx::Accessor> &accessors, const T &rc){
        cytnx_error_msg(accessors.size() > this->_shape.size(), "%s", "The input indexes rank is out of range! (>Tensor's rank).");

        vector<cytnx::Accessor> acc = accessors;
        for(int i=0;i<this->_shape.size()-accessors.size();i++){
            acc.push_back(Accessor::all());    
        }

        vector<cytnx_uint64> get_shape(acc.size());
        //vector<cytnx_uint64> new_shape;

        std::vector<std::vector<cytnx_uint64> > locators(acc.size());
        for(cytnx_uint32 i=0;i<acc.size();i++){
            acc[i].get_len_pos(this->_shape[i],get_shape[i],locators[i]); 
            //std::cout << this->_shape[i] << " " << get_shape[i] << "|";
            //for(int j=0;j<locators[i].size();j++) std::cout << locators[i][j] << " ";
            //std::cout << std::endl;
        }   

        //remove single dim
        vector<cytnx_uint64> new_shape;
        for(cytnx_uint32 i=0;i<acc.size();i++)
            if(get_shape[i]!=1) new_shape.push_back(get_shape[i]);

        if(new_shape.size()==0) new_shape.push_back(1);
        
        //boost::intrusive_ptr<Tensor_impl> out( new Tensor_impl());
        //out->Init(get_shape,this->dtype(),this->device());

        Storage tmp(1,Type.c_typename_to_id(typeid(T).name()),this->device());
        tmp.at<T>(0) = rc;
        this->storage()._impl->SetElem_byShape(tmp._impl,this->shape(),this->_mapper,get_shape,locators,true);
    }
    template void Tensor_impl::set<cytnx_complex128>(const std::vector<cytnx::Accessor> &, const cytnx_complex128&);
    template void Tensor_impl::set<cytnx_complex64>(const std::vector<cytnx::Accessor> &, const cytnx_complex64&);
    template void Tensor_impl::set<cytnx_double>(const std::vector<cytnx::Accessor> &, const cytnx_double&);
    template void Tensor_impl::set<cytnx_float>(const std::vector<cytnx::Accessor> &, const cytnx_float&);
    template void Tensor_impl::set<cytnx_int64>(const std::vector<cytnx::Accessor> &, const cytnx_int64&);
    template void Tensor_impl::set<cytnx_uint64>(const std::vector<cytnx::Accessor> &, const cytnx_uint64&);
    template void Tensor_impl::set<cytnx_int32>(const std::vector<cytnx::Accessor> &, const cytnx_int32&);
    template void Tensor_impl::set<cytnx_uint32>(const std::vector<cytnx::Accessor> &, const cytnx_uint32&);
    template void Tensor_impl::set<cytnx_int16>(const std::vector<cytnx::Accessor> &, const cytnx_int16&);
    template void Tensor_impl::set<cytnx_uint16>(const std::vector<cytnx::Accessor> &, const cytnx_uint16&);
    template void Tensor_impl::set<cytnx_bool>(const std::vector<cytnx::Accessor> &, const cytnx_bool&);






    std::ostream& operator<<(std::ostream& os,const Tensor &in){
        if(in.is_contiguous()) in._impl->storage()._impl->PrintElem_byShape(os,in.shape());
        else in._impl->storage()._impl->PrintElem_byShape(os,in.shape(),in._impl->invmapper());
        return os;
    }       
    std::ostream& operator<<(std::ostream& os,const Tensor::Tproxy &in){
        os << Tensor(in) << std::endl;
        return os;
    }       
    //===================================================================
    //wrapper

    void Tensor::Tofile(const std::string &fname) const{
        if(!this->is_contiguous()){
            auto A = this->contiguous();
            A.storage().Tofile(fname);  
        }else{
            this->_impl->_storage.Tofile(fname);
        }
    }
    void Tensor::Tofile(const char* fname) const{
        if(!this->is_contiguous()){
            auto A = this->contiguous();
            A.storage().Tofile(fname);  
        }else{
            this->_impl->_storage.Tofile(fname);
        }
    }
    void Tensor::Tofile(fstream &f) const{

        if(!this->is_contiguous()){
            auto A = this->contiguous();
            A.storage().Tofile(f);  
        }else{
            this->_impl->_storage.Tofile(f);
        }

    }
    void Tensor::Save(const std::string &fname) const{
        fstream f;
        f.open((fname+".cytn"),ios::out|ios::trunc|ios::binary);
        if(!f.is_open()){
            cytnx_error_msg(true,"[ERROR] invalid file path for save.%s","\n");
        }
        this->_Save(f);
        f.close();
    }
    void Tensor::Save(const char* fname) const{
        fstream f;
        string ffname = string(fname) + ".cytn";
        f.open(ffname,ios::out|ios::trunc|ios::binary);
        if(!f.is_open()){
            cytnx_error_msg(true,"[ERROR] invalid file path for save.%s","\n");
        }
        this->_Save(f);
        f.close();
    }
    void Tensor::_Save(fstream &f) const{
        //header
        //check:
        cytnx_error_msg(!f.is_open(),"[ERROR] invalid fstream!.%s","\n");

        unsigned int IDDs = 888;
        f.write((char*)&IDDs,sizeof(unsigned int));
        cytnx_uint64 shp = this->shape().size();
        cytnx_uint64 Conti = this->is_contiguous();
        f.write((char*)&shp,sizeof(cytnx_uint64));

        f.write((char*)&Conti,sizeof(cytnx_uint64));
        f.write((char*)&this->_impl->_shape[0],sizeof(cytnx_uint64)*shp);
        f.write((char*)&this->_impl->_mapper[0],sizeof(cytnx_uint64)*shp);
        f.write((char*)&this->_impl->_invmapper[0],sizeof(cytnx_uint64)*shp);

        //pass to storage for save:
        this->_impl->_storage._Save(f);

    }



    
    Tensor Tensor::Fromfile(const std::string &fname, const unsigned int &dtype,const cytnx_int64 &count){
        return Tensor::from_storage(Storage::Fromfile(fname,dtype,count)); 
    }
    Tensor Tensor::Fromfile(const char* fname, const unsigned int &dtype,const cytnx_int64 &count){
        return Tensor::from_storage(Storage::Fromfile(fname,dtype,count)); 
    }
    Tensor Tensor::Load(const std::string &fname){
        Tensor out;
        fstream f;
        f.open(fname,ios::in|ios::binary);
        if(!f.is_open()){
            cytnx_error_msg(true,"[ERROR] invalid file path for load.%s","\n");
        }
        out._Load(f);
        f.close();
        return out;
    }
    Tensor Tensor::Load(const char* fname){
        Tensor out;
        fstream f;
        f.open(fname,ios::in|ios::binary);
        if(!f.is_open()){
            cytnx_error_msg(true,"[ERROR] invalid file path for load.%s","\n");
        }
        out._Load(f);
        f.close();
        return out;
    }
    void Tensor::_Load(fstream &f){
        
        //header
        //check:
        cytnx_error_msg(!f.is_open(),"[ERROR] invalid fstream!.%s","\n");

        unsigned int tmpIDDs;
        f.read((char*)&tmpIDDs,sizeof(unsigned int));
        cytnx_error_msg(tmpIDDs!=888,"[ERROR] the object is not a cytnx tensor!%s","\n");

        cytnx_uint64 shp  ;
        cytnx_uint64 Conti; 
        f.read((char*)&shp,sizeof(cytnx_uint64));
        f.read((char*)&Conti,sizeof(cytnx_uint64));
        this->_impl->_contiguous = Conti;


        this->_impl->_shape.resize(shp);
        this->_impl->_mapper.resize(shp);
        this->_impl->_invmapper.resize(shp);
        f.read((char*)&this->_impl->_shape[0],sizeof(cytnx_uint64)*shp);
        f.read((char*)&this->_impl->_mapper[0],sizeof(cytnx_uint64)*shp);
        f.read((char*)&this->_impl->_invmapper[0],sizeof(cytnx_uint64)*shp);

        //pass to storage for save:
        this->_impl->_storage._Load(f);
    }


    Tensor Tensor::real(){
        Tensor out; 
        out._impl = this->_impl->_clone_meta_only();
        out._impl->_storage = this->_impl->_storage.real();
        return out;
    };

    Tensor Tensor::imag(){
        Tensor out; 
        out._impl = this->_impl->_clone_meta_only();
        out._impl->_storage = this->_impl->_storage.imag();
        return out;
    }


    // += 
    template<> Tensor& Tensor::operator+=<Tensor>(const Tensor &rc){
        this->_impl->storage() = cytnx::linalg::Add(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator+=<Tensor::Tproxy>(const Tensor::Tproxy &rc){
        this->_impl->storage() = cytnx::linalg::Add(*this,Tensor(rc))._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator+=<cytnx_complex128>(const cytnx_complex128 &rc){
        this->_impl->storage() = cytnx::linalg::Add(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator+=<cytnx_complex64>(const cytnx_complex64 &rc){
        this->_impl->storage() = cytnx::linalg::Add(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator+=<cytnx_double>(const cytnx_double &rc){
        this->_impl->storage() = cytnx::linalg::Add(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator+=<cytnx_float>(const cytnx_float &rc){
        this->_impl->storage() = cytnx::linalg::Add(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator+=<cytnx_int64>(const cytnx_int64 &rc){
        this->_impl->storage() = cytnx::linalg::Add(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator+=<cytnx_uint64>(const cytnx_uint64 &rc){
        this->_impl->storage() = cytnx::linalg::Add(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator+=<cytnx_int32>(const cytnx_int32 &rc){
        this->_impl->storage() = cytnx::linalg::Add(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator+=<cytnx_uint32>(const cytnx_uint32 &rc){
        this->_impl->storage() = cytnx::linalg::Add(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator+=<cytnx_int16>(const cytnx_int16 &rc){
        this->_impl->storage() = cytnx::linalg::Add(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator+=<cytnx_uint16>(const cytnx_uint16 &rc){
        this->_impl->storage() = cytnx::linalg::Add(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator+=<cytnx_bool>(const cytnx_bool &rc){
        this->_impl->storage() = cytnx::linalg::Add(*this,rc)._impl->storage();
        return *this;
    }

    // -= 
    template<> Tensor& Tensor::operator-=<Tensor>(const Tensor &rc){
        this->_impl->storage() = cytnx::linalg::Sub(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator-=<Tensor::Tproxy>(const Tensor::Tproxy &rc){
        this->_impl->storage() = cytnx::linalg::Sub(*this,Tensor(rc))._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator-=<cytnx_complex128>(const cytnx_complex128 &rc){
        this->_impl->storage() = cytnx::linalg::Sub(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator-=<cytnx_complex64>(const cytnx_complex64 &rc){
        this->_impl->storage() = cytnx::linalg::Sub(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator-=<cytnx_double>(const cytnx_double &rc){
        this->_impl->storage() = cytnx::linalg::Sub(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator-=<cytnx_float>(const cytnx_float &rc){
        this->_impl->storage() = cytnx::linalg::Sub(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator-=<cytnx_int64>(const cytnx_int64 &rc){
        this->_impl->storage() = cytnx::linalg::Sub(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator-=<cytnx_uint64>(const cytnx_uint64 &rc){
        this->_impl->storage() = cytnx::linalg::Sub(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator-=<cytnx_int32>(const cytnx_int32 &rc){
        this->_impl->storage() = cytnx::linalg::Sub(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator-=<cytnx_uint32>(const cytnx_uint32 &rc){
        this->_impl->storage() = cytnx::linalg::Sub(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator-=<cytnx_int16>(const cytnx_int16 &rc){
        this->_impl->storage() = cytnx::linalg::Sub(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator-=<cytnx_uint16>(const cytnx_uint16 &rc){
        this->_impl->storage() = cytnx::linalg::Sub(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator-=<cytnx_bool>(const cytnx_bool &rc){
        this->_impl->storage() = cytnx::linalg::Sub(*this,rc)._impl->storage();
        return *this;
    }

    // *= 
    template<> Tensor& Tensor::operator*=<Tensor>(const Tensor &rc){
        this->_impl->storage() = cytnx::linalg::Mul(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator*=<Tensor::Tproxy>(const Tensor::Tproxy &rc){
        this->_impl->storage() = cytnx::linalg::Mul(*this,Tensor(rc))._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator*=<cytnx_complex128>(const cytnx_complex128 &rc){
        this->_impl->storage() = cytnx::linalg::Mul(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator*=<cytnx_complex64>(const cytnx_complex64 &rc){
        this->_impl->storage() = cytnx::linalg::Mul(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator*=<cytnx_double>(const cytnx_double &rc){
        this->_impl->storage() = cytnx::linalg::Mul(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator*=<cytnx_float>(const cytnx_float &rc){
        this->_impl->storage() = cytnx::linalg::Mul(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator*=<cytnx_int64>(const cytnx_int64 &rc){
        this->_impl->storage() = cytnx::linalg::Mul(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator*=<cytnx_uint64>(const cytnx_uint64 &rc){
        this->_impl->storage() = cytnx::linalg::Mul(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator*=<cytnx_int32>(const cytnx_int32 &rc){
        this->_impl->storage() = cytnx::linalg::Mul(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator*=<cytnx_uint32>(const cytnx_uint32 &rc){
        this->_impl->storage() = cytnx::linalg::Mul(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator*=<cytnx_int16>(const cytnx_int16 &rc){
        this->_impl->storage() = cytnx::linalg::Mul(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator*=<cytnx_uint16>(const cytnx_uint16 &rc){
        this->_impl->storage() = cytnx::linalg::Mul(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator*=<cytnx_bool>(const cytnx_bool &rc){
        this->_impl->storage() = cytnx::linalg::Mul(*this,rc)._impl->storage();
        return *this;
    }

    // /= 
    template<> Tensor& Tensor::operator/=<Tensor>(const Tensor &rc){
        this->_impl->storage() = cytnx::linalg::Div(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator/=<Tensor::Tproxy>(const Tensor::Tproxy &rc){
        this->_impl->storage() = cytnx::linalg::Div(*this,Tensor(rc))._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator/=<cytnx_complex128>(const cytnx_complex128 &rc){
        this->_impl->storage() = cytnx::linalg::Div(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator/=<cytnx_complex64>(const cytnx_complex64 &rc){
        this->_impl->storage() = cytnx::linalg::Div(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator/=<cytnx_double>(const cytnx_double &rc){
        this->_impl->storage() = cytnx::linalg::Div(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator/=<cytnx_float>(const cytnx_float &rc){
        this->_impl->storage() = cytnx::linalg::Div(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator/=<cytnx_int64>(const cytnx_int64 &rc){
        this->_impl->storage() = cytnx::linalg::Div(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator/=<cytnx_uint64>(const cytnx_uint64 &rc){
        this->_impl->storage() = cytnx::linalg::Div(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator/=<cytnx_int32>(const cytnx_int32 &rc){
        //std::cout << "entry /= int32" << std::endl;
        this->_impl->storage() = cytnx::linalg::Div(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator/=<cytnx_uint32>(const cytnx_uint32 &rc){
        this->_impl->storage() = cytnx::linalg::Div(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator/=<cytnx_int16>(const cytnx_int16 &rc){
        this->_impl->storage() = cytnx::linalg::Div(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator/=<cytnx_uint16>(const cytnx_uint16 &rc){
        this->_impl->storage() = cytnx::linalg::Div(*this,rc)._impl->storage();
        return *this;
    }
    template<> Tensor& Tensor::operator/=<cytnx_bool>(const cytnx_bool &rc){
        this->_impl->storage() = cytnx::linalg::Div(*this,rc)._impl->storage();
        return *this;
    }

    std::vector<Tensor> Tensor::Svd(const bool&is_U, const bool&is_vT) const{
        return linalg::Svd(*this, is_U, is_vT);
    }
    std::vector<Tensor> Tensor::Eigh(const bool &is_V,const bool &row_v) const{
        return linalg::Eigh(*this, is_V,row_v);
    }


    Tensor& Tensor::InvM_(){
        linalg::InvM_(*this);
        return *this;
    }
    Tensor Tensor::InvM() const{
        return linalg::InvM(*this); 
    }
    Tensor& Tensor::Inv_(const double &clip){
        linalg::Inv_(*this,clip);
        return *this;
    }
    Tensor Tensor::Inv(const double &clip) const{
        return linalg::Inv(*this,clip); 
    }

    
    Tensor& Tensor::Conj_(){
        linalg::Conj_(*this);
        return *this;
    }
    Tensor Tensor::Conj() const{
        return linalg::Conj(*this); 
    }

    Tensor& Tensor::Exp_(){
        linalg::Exp_(*this);
        return *this;
    }
    Tensor Tensor::Exp() const{
        return linalg::Exp(*this); 
    }
    Tensor Tensor::Norm() const{
        return linalg::Norm(*this);
    }    

    Tensor Tensor::Pow(const cytnx_double &p) const{
        return linalg::Pow(*this,p);
    }    

    Tensor& Tensor::Pow_(const cytnx_double &p){
        linalg::Pow_(*this,p);
        return *this;
    }

    Tensor& Tensor::Abs_(){
        linalg::Abs_(*this);
        return *this;
    }
    Tensor Tensor::Abs() const{
        return linalg::Abs(*this);
    }
    Tensor Tensor::Max() const{
        return linalg::Max(*this);
    }
    Tensor Tensor::Min() const{
        return linalg::Min(*this);
    }

    Tensor Tensor::Trace(const cytnx_uint64 &a, const cytnx_uint64 &b) const{
        Tensor out = linalg::Trace(*this,a,b);
        return out;
    }

    bool Tensor::same_data(const Tensor &rhs)const{
        return is(this->_impl->storage(),rhs.storage());
    }

    //===========================
    //Tensor am Tproxy
    Tensor operator+(const Tensor &lhs, const Tensor::Tproxy &rhs){
        return cytnx::linalg::Add(lhs,Tensor(rhs));
    }
    Tensor operator-(const Tensor &lhs, const Tensor::Tproxy &rhs){
        return cytnx::linalg::Sub(lhs,Tensor(rhs));
    }
    Tensor operator*(const Tensor &lhs, const Tensor::Tproxy &rhs){
        return cytnx::linalg::Mul(lhs,Tensor(rhs));
    }
    Tensor operator/(const Tensor &lhs, const Tensor::Tproxy &rhs){
        return cytnx::linalg::Div(lhs,Tensor(rhs));
    }


}//namespace cytnx


