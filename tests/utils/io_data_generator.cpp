// Generator for the committed HDF5 reference data used by the io load tests.
//
// This file does not contain real tests; it only (re)generates the reference
// files in "tests/test_data_base/io" using io::Save, so that the io load tests
// can verify that previously-saved files remain readable (format-stability /
// regression coverage).
//
// To (re)generate the reference data, set NEED_GEN_IO_DATA to 1, rebuild, and
// run the test binary once (e.g. `./test_main --gtest_filter='IODataGen.*'`).
// Then set NEED_GEN_IO_DATA back to 0 and commit the generated files.
#define NEED_GEN_IO_DATA 0
#if NEED_GEN_IO_DATA

  #include <filesystem>

  #include "cytnx.hpp"
  #include "io.hpp"
  #include "io_test/io_test_tools.h"

using namespace cytnx;
using namespace cytnx::IOTest;

namespace {
  void ensure_dir() { std::filesystem::create_directories(ref_data_dir()); }
}  // namespace

TEST(IODataGen, GenerateAll) {
  ensure_dir();
  SaveToFile(ref::storage(), ref_data_dir() + "storage.h5");
  SaveToFile(ref::tensor(), ref_data_dir() + "tensor.h5");
  SaveToFile(ref::bond(), ref_data_dir() + "bond.h5");
  SaveToFile(ref::bond_fermionic(), ref_data_dir() + "bond_fermionic.h5");
  SaveToFile(ref::symmetry(), ref_data_dir() + "symmetry.h5");
  SaveToFile(ref::symmetry_fpar(), ref_data_dir() + "symmetry_fpar.h5");
  SaveToFile(ref::unitensor_dense(), ref_data_dir() + "unitensor_dense.h5");
  SaveToFile(ref::unitensor_diag(), ref_data_dir() + "unitensor_diag.h5");
  SaveToFile(ref::unitensor_sym(), ref_data_dir() + "unitensor_sym.h5");
  SaveToFile(ref::unitensor_fermionic(), ref_data_dir() + "unitensor_fermionic.h5");
}

#endif  // NEED_GEN_IO_DATA
