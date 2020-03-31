#include "random_internal_interface.hpp"
#include <vector>

using namespace std;
namespace cytnx{
    namespace random_internal{

        random_internal_interface::random_internal_interface(){



            /// function signature. 
            Normal = vector<Rnd_io>(N_Type,NULL);
            Normal[Type.ComplexDouble] = Rng_normal_cpu_cd;
            Normal[Type.ComplexFloat ] = Rng_normal_cpu_cf;
            Normal[Type.Double       ] = Rng_normal_cpu_d;
            Normal[Type.Float        ] = Rng_normal_cpu_f;


            #ifdef UNI_GPU
                cuNormal = vector<Rnd_io>(N_Type,NULL);
                cuNormal[Type.ComplexDouble] = cuRng_normal_cd;
                cuNormal[Type.ComplexFloat ] = cuRng_normal_cf;
                cuNormal[Type.Double       ] = cuRng_normal_d;
                cuNormal[Type.Float        ] = cuRng_normal_f;
            #endif


        }

        random_internal_interface rii;

    }
}


