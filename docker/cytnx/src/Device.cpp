#include "Device.hpp"
#include "cytnx_error.hpp"

using namespace std;
namespace cytnx {
  Device_class::Device_class() : Ngpus(0) {
#ifdef UNI_GPU
    // get all available gpus
    checkCudaErrors(cudaGetDeviceCount(&Ngpus));

    CanAccessPeer = vector<vector<bool>>(Ngpus, vector<bool>(Ngpus));
    // check can Peer Access, if can, open PCIE access to increase bandwidth
    //  Enable Peer Access when it's possible:
    //  https://stackoverflow.com/questions/31628041/how-to-copy-memory-between-different-gpus-in-cuda
    int cAP;
    vector<bool> isopen(Ngpus);
    for (int i = 0; i < Ngpus; i++) {
      CanAccessPeer[i][i] = 1;
      for (int j = i + 1; j < Ngpus; j++) {
        cudaDeviceCanAccessPeer(&cAP, i, j);
        if (cAP && !isopen[i]) {
          cudaDeviceEnablePeerAccess(i, 0);
          isopen[i] = 1;
        }
        if (cAP && !isopen[j]) {
          cudaDeviceEnablePeerAccess(j, 0);
          isopen[j] = 1;
        }
        if (cAP) {
          CanAccessPeer[i][j] = 1;
          CanAccessPeer[j][i] = 1;
        }
      }
    }
#endif
  };
  string Device_class::getname(const int &device_id) {
    if (device_id == this->cpu) {
      return string("cytnx device: CPU");
    } else if (device_id >= 0) {
      if (device_id >= Ngpus) {
        cytnx_error_msg(true, "%s", "[ERROR] invalid device_id, gpuid exceed limit");
      } else {
        return string("cytnx device: CUDA/GPU-id:") + to_string(device_id);
      }
    } else {
      cytnx_error_msg(true, "%s", "[ERROR] invalid device_id");
    }
  }
  void Device_class::Print_Property() {
#ifdef UNI_GPU
    cout << "=== CUDA support ===" << endl;
    cout << ": Peer PCIE Access:" << endl;
    cout << "   ";
    for (int i = 0; i < this->Ngpus; i++) printf(" %2d", i);
    cout << endl;

    cout << "   ";
    for (int i = 0; i < this->Ngpus; i++) printf("%s", "---");
    cout << endl;

    for (int i = 0; i < this->Ngpus; i++) {
      printf("%2d|", i);
      for (int j = 0; j < this->Ngpus; j++) {
        if (j == i)
          printf("%s", "  x");
        else
          printf("  %d", int(CanAccessPeer[i][j]));
      }
      cout << endl;
    }
    cout << "--------------------" << endl;
#else
    cout << "=== No CUDA support ===" << endl;
#endif
  }

  Device_class Device;
}  // namespace cytnx

// Maybe handle GPU part here.
