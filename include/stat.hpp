#ifndef _stat_H_
#define _stat_H_

#include "Type.hpp"
#include "cytnx_error.hpp"
#include "Storage.hpp"
#include "Tensor.hpp"
#include <algorithm>
#include <iostream>
namespace cytnx{
    namespace stat{
        
        /// 1D, real value histogram 
        class Histogram{
            public:
                double min;
                double max;
                uint64_t bins;
                cytnx::Storage vars;
                cytnx::Storage x;

                //std::vector<double> vars;
                //std::vector<double> x;

                double total_count;
                
               
                /**
                @brief initialize a histogram 
                */            
                Histogram(const unsigned long long &Nbins, const double &min_val, const double &max_val);
               
                ///@cond
                Histogram(const Histogram &rhs){
                    this->min = rhs.min;
                    this->max = rhs.max;
                    this->bins = rhs.bins;
                    this->vars = rhs.vars.clone();
                    this->x    = rhs.x.clone();
                    this->total_count = rhs.total_count;
                }

                Histogram& operator=(const Histogram &rhs){
                    this->min = rhs.min;
                    this->max = rhs.max;
                    this->bins = rhs.bins;
                    this->vars = rhs.vars.clone();
                    this->x    = rhs.x.clone();
                    this->total_count = rhs.total_count;    
                    return *this;
                }
                ///@endcond
     
                void clear_vars(){
                    total_count = 0;
                    memset(this->vars.data(),0,sizeof(double)*this->vars.size());
                }        

                template<class T>
                void accumulate(const std::vector<T> &data){
                    std::vector<T> tmp = data;
                    std::sort(tmp.begin(),tmp.end());

                    uint64_t cntr = 0;
                    double curr_x = 0;
                    double dx = double(max-min)/bins;

                    // for each elem in data, compare
                    for(unsigned long long i=0;i<tmp.size();i++){
                        while(cntr<=this->bins){
                            if(tmp[i] < curr_x){
                                if(cntr){ vars.at<double>(cntr-1) += 1; total_count+=1;}
                                break;
                            }   
                            cntr++;
                            curr_x = cntr*dx;
                        }
                    }    
                }

                void normalize();
                void print() const;                
                
                const Storage& get_x()const{
                    //get x 
                    return this->x;
                }
                Storage& get_x(){
                    //get x 
                    return this->x;
                }
                
                cytnx_uint64 size(){
                    return this->x.size();
                }

        };



    }//stat
}//cytnx


#endif
