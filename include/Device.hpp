#ifndef _H_Device_
#define _H_Device_
#include <vector>
#include <string>

struct __device{
    enum __pybind_device{
        cpu=-1,
        cuda=0
    };
};

namespace tor10{
    class Device{
        public:
            enum:int{
                cpu=-1,
                cuda=0
            };
            int Ngpus;
            std::vector<std::vector<bool> > CanAccessPeer;
            Device();
            void Print_Property();
            std::string getname(const int &device_id);
    };
    extern Device tor10device;
}//namespace tor10
#endif
