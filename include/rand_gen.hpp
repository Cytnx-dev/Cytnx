#ifndef _rand_gen_H_
#define _rand_gen_H_

#include "Type.hpp"
#include "intrusive_ptr_base.hpp"
#include "cytnx_error.hpp"
#include <random>
#include <string>


namespace cytnx{
    namespace random{

    class generator_base: public intrusive_ptr_base<generator_base>{
        private:
        public:
            friend class generator;
            unsigned int _seed;
            unsigned int _engtype;

            generator_base(){
                this->_engtype = cytnx::EngType.Non;
            }

            bool is_cuda() const{
                return EngType.is_cuda(this->_engtype);
            }
            bool is_cpu() const{
                return EngType.is_cpu(this->_engtype);
            }
            const unsigned int &engtype() const{
                return this->_engtype;
            }
            const unsigned int &seed() const{
                return this->_seed;
            }
            
            virtual void set_seed(const unsigned int &seed);


    };

    class mt19937_Eng: public generator_base{
        public:
            std::mt19937 _gen;
            mt19937_Eng(const unsigned int &seed){this->_engtype=EngType.Mt19937; this->_seed = seed; this->_gen = std::mt19937(seed);};
            
            void set_seed(const unsigned int &seed){
                this->_gen = std::mt19937(seed);
                this->_seed = seed;
            }

    };


    class mt19937_64_Eng: public generator_base{
        public:
            std::mt19937_64 _gen;
            mt19937_64_Eng(const unsigned int &seed){this->_engtype=EngType.Mt19937_64; this->_seed = seed; this->_gen=std::mt19937_64(seed);};
            
            void set_seed(const unsigned int &seed){
                this->_gen = std::mt19937_64(seed);
                this->_seed = seed;
            }
    };


    class generator{
        public:
            boost::intrusive_ptr<generator_base> _impl;
            generator(): _impl(new generator_base()){};
            generator(const generator &rhs){
                _impl = rhs._impl;
            }
            generator& operator=(const generator &rhs){
                _impl = rhs._impl;
                return *this;
            }

            void Init(const unsigned int &type, const unsigned int &seed=std::random_device()()){
                
                //dyn disp. 
                if(type==EngType.Mt19937){
                  this->_impl = boost::intrusive_ptr<generator_base>(new mt19937_Eng(seed)); 
                }else if(type==EngType.Mt19937_64){
                  this->_impl = boost::intrusive_ptr<generator_base>(new mt19937_64_Eng(seed)); 
                }else{
                    cytnx_error_msg(true,"[random::generator] Invalid engine type.%s","\n");
                }
                
            }
   
            const unsigned int &seed() const{
                return this->_impl->seed();
            }
            const void set_seed(const unsigned int &seed){
                this->_impl->set_seed(seed);
            }
            const unsigned int &engtype()const{
                return this->_impl->engtype();
            }
            const std::string &engtype_str() const{
                return EngType.getname(this->_impl->engtype());
            }
            const unsigned int word_size() const{
                return EngType.word_size(this->_impl->engtype());
            }
            const bool is_cuda() const{
                return this->_impl->is_cuda();
            }
            const bool is_cpu() const{
                return this->_impl->is_cpu();
            }
    };

    }//random
}// cytnx

#endif
