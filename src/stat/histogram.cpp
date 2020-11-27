#include "stat.hpp"
#include "Type.hpp"

namespace cytnx{
    namespace stat{

        Histogram::Histogram(const unsigned long long &Nbins, const double &min_val, const double &max_val){

            if(min_val >= max_val){
                std::cout << "[ERROR] cannot have min >= max" << std::endl;exit(1);
            }
            this->min = min_val;
            this->max = max_val;
            this->bins = Nbins;

            this->vars = Storage(bins,Type.Double);
            this->x = Storage(bins,Type.Double);
            double dx = double(max_val-min_val)/Nbins;
            for(unsigned int i=0;i<x.size();i++){
                this->x.at<double>(i) = dx*i;
            }

            total_count = 0;

        }

        void Histogram::normalize(){
            //get the density. 
            double dx = double(max-min)/bins;
            double w = 1./total_count/dx;

            //wrapping around storage and use Tensor API
            Tensor tmp = Tensor::from_storage(this->vars);
            tmp *= w;
            this->vars = tmp.storage();


        }

        void Histogram::print() const{
            std::cout << "[Histogram 1D] Real" << std::endl;
            std::cout << "Nbins: " << this->bins << std::endl;
            std::cout << "bound: [ " << this->min << " , " << this->max << " ]\n";
            std::cout << "current count: " << this->total_count << std::endl;
        }


    }//stat
}//cytnx




