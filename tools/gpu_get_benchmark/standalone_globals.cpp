// Minimal Cytnx global required by cytnx_error.hpp in the standalone build.
// The full Cytnx library defines this in src/Type.cpp; do not link both.
namespace cytnx {
bool User_debug = false;
}
