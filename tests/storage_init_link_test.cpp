#include "cytnx.hpp"

// Issue #673: only use the inline Storage constructor and accessors here.
// In the old implementation, linking this minimal executable against the
// static library could omit Storage.cpp (and its __SII initializer), causing
// Storage::Init to call a null function pointer. Do not add operator<< or
// link this test into the larger test_main executable.
int main() {
  const cytnx::Storage storage(10);
  return storage.size() == 10 && storage.dtype() == cytnx::Type.Double ? 0 : 1;
}
