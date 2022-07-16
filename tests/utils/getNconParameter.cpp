#include "getNconParameter.h"
int t, n, m, b;
vector<vector<cytnx_int64>> bonds;
vector<vector<cytnx_int64>> links;
vector<UniTensor> uTs;
pair<vector<UniTensor>, vector<vector<cytnx_int64>>> getNconParameter(string file) {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  ifstream cin(file);

  cin >> n;
  rep(i, 0, n) {
    vector<cytnx_int64> bond;
    cin >> b;
    rep(j, 0, b) {
      cin >> t;
      bond.pb(t);
    }
    vector<cytnx_uint64> ubond;
    transform(all(bond), back_inserter(ubond), [](cytnx_int64 x) { return (cytnx_uint64)x; });
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
    UniTensor uT = UniTensor(T, 0);
    uTs.pb(uT);
  }
  rep(i, 0, n) {
    vector<cytnx_int64> link;
    rep(j, 0, SZ(bonds[i])) {
      cin >> t;
      link.pb(t);
    }
    links.pb(link);
  }
  return {uTs, links};
  //   UniTensor res = ncon(uTs, links);
  //   int tot_dim = 1;

  //   vector<cytnx_uint64> shape = res.shape();
  //   rep(i, 0, SZ(shape)) tot_dim *= shape[i];
  //   vector<cytnx_int64> vtot_dim;
  //   vtot_dim.pb(tot_dim);
  //   res.reshape_(vtot_dim);
  //   rep(i, 0, tot_dim) cout << res.get_block()(i).item() << ' ';
  //   cout << '\n';
}
