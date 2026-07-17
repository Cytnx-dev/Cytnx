#include "getNconParameter.h"

namespace cytnx {
  namespace test {

// #define int long long
#define rep(i, a, n) for (int i = a; i < n; i++)
#define per(i, a, n) for (int i = n - 1; i >= a; i--)
#define pb push_back
// #define mp std::make_pair
#define all(x) (x).begin(), (x).end()
#define fi first
#define se second
#define SZ(x) ((int)(x).size())
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))
#define abs(x) (((x) < 0) ? (-(x)) : (x))
    std::mt19937 mrand(std::random_device{}());
    static int rnd(int x) { return mrand() % x; }
    // head
    int t, n, m, b;
    std::vector<std::vector<cytnx_int64>> bonds;
    std::vector<std::vector<cytnx_int64>> links;
    std::vector<UniTensor> uTs;
    std::pair<std::vector<UniTensor>, std::vector<std::vector<cytnx_int64>>> getNconParameter(
      std::string file) {
      std::ios_base::sync_with_stdio(0);
      std::cin.tie(0);
      std::cout.tie(0);
      std::ifstream cin(file);

      cin >> n;
      rep(i, 0, n) {
        std::vector<cytnx_int64> bond;
        cin >> b;
        rep(j, 0, b) {
          cin >> t;
          bond.pb(t);
        }
        std::vector<cytnx_uint64> ubond;
        std::transform(all(bond), std::back_inserter(ubond),
                       [](cytnx_int64 x) { return (cytnx_uint64)x; });
        bonds.pb(bond);
        Tensor T = zeros(ubond);
        int tot_dim = 1;
        rep(j, 0, SZ(bond)) tot_dim *= bond[j];
        T.reshape_(tot_dim);
        rep(j, 0, tot_dim) {
          cin >> t;
          T(j) = t;
        }
        T.reshape_(bond);
        UniTensor uT =
          UniTensor(T, false, 0);  // The UniTensors in output.txt are rowrank zero UniTensors
        uTs.pb(uT);
      }
      rep(i, 0, n) {
        std::vector<cytnx_int64> link;
        rep(j, 0, SZ(bonds[i])) {
          cin >> t;
          link.pb(t);
        }
        links.pb(link);
      }
      return {uTs, links};
      //   UniTensor res = ncon(uTs, links);
      //   int tot_dim = 1;

      //   std::vector<cytnx_uint64> shape = res.shape();
      //   rep(i, 0, SZ(shape)) tot_dim *= shape[i];
      //   std::vector<cytnx_int64> vtot_dim;
      //   vtot_dim.pb(tot_dim);
      //   res.reshape_(vtot_dim);
      //   rep(i, 0, tot_dim) std::cout << res.get_block()(i).item() << ' ';
      //   std::cout << '\n';
    }
  }  // namespace test
}  // namespace cytnx
