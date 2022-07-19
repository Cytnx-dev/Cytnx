#include <bits/stdc++.h>
using namespace std;
#define rep(i, a, n) for (int i = a; i < n; i++)
#define per(i, a, n) for (int i = n - 1; i >= a; i--)
#define pb push_back
#define all(x) (x).begin(), (x).end()
#define fi first
#define se second
#define SZ(x) ((int)(x).size())
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))
#define abs(x) (((x) < 0) ? (-(x)) : (x))
typedef vector<int> VI;
typedef long long ll;
typedef pair<int, int> PII;
typedef double db;
mt19937 mrand(random_device{}());
int rnd(int x) { return mrand() % x; }
// head
/*
 * This program will generate random networks(of certain format) into output.txt
 * _n is the max tensor count
 * _d is the max bond count per tensor
 * _num is the upperbound of randomly generated numbers
 */
const int _n = 20, _d = 5, _num = 10;
int t, n, tot_bond_cnt = 0;
VI flatlink, flatbonds;
vector<VI> bonds;
int main(void) {
  ofstream cout("output.txt");
  n = rnd(_n) + 1;
  cout << n << '\n';
  rep(_, 0, n) {
    int d = rnd(_d) + 1;
    int dd = 0;
    VI l;
    while (dd != d) {
      int li = rnd(d - dd) + 1;
      dd += li;
      l.pb(li);
      flatbonds.pb(li);
    }
    bonds.pb(l);
    tot_bond_cnt += SZ(l);
  }
  // At most 3*2=6 uncontracted bond for preventing from explosion dimention of tensor
  int tmp = rnd((int)tot_bond_cnt / ((int)2));
  int cont_cnt = max(tmp, tot_bond_cnt / 2 - 3), neg_bond_cnt = tot_bond_cnt - 2 * cont_cnt;
  rep(i, 1, cont_cnt + 1) rep(__, 0, 2) flatlink.pb(i);
  rep(i, 1, neg_bond_cnt + 1) flatlink.pb(-i);
  shuffle(all(flatlink), mrand);

  VI last_cont_idx(cont_cnt + 1, -1);
  per(i, 0, tot_bond_cnt) if (flatlink[i] > 0 and last_cont_idx[flatlink[i]] == -1)
    last_cont_idx[flatlink[i]] = i;

  int idx0 = 0;
  rep(i, 0, SZ(bonds)) {
    int ele_cnt = 1;
    rep(j, 0, SZ(bonds[i])) {
      if (flatlink[idx0 + j] > 0 and last_cont_idx[flatlink[idx0 + j]] != idx0 + j)
        flatbonds[idx0 + j] = flatbonds[last_cont_idx[flatlink[idx0 + j]]];
      bonds[i][j] = flatbonds[idx0 + j];
      ele_cnt *= bonds[i][j];
    }

    idx0 += SZ(bonds[i]);

    cout << SZ(bonds[i]) << '\n';
    rep(j, 0, SZ(bonds[i])) {
      cout << bonds[i][j];
      if (j != SZ(bonds[i]) - 1) cout << ' ';
    }
    cout << '\n';
    rep(j, 0, ele_cnt) {
      cout << rnd(_num) + 1;  // Do not generate zero
      if (j != ele_cnt - 1) cout << ' ';
    }
    cout << '\n';
  }

  vector<VI> links;
  int idx = 0;
  rep(i, 0, SZ(bonds)) {
    VI link;
    rep(j, 0, SZ(bonds[i])) link.pb(flatlink[idx + j]);
    idx += SZ(bonds[i]);
    links.pb(link);
  }

  // ENSURE NO PARTIAL TRACE PRESENTS
  VI pos(cont_cnt + 1, -1);
  rep(i, 0, SZ(links)) {
    rep(j, 0, SZ(links[i])) {
      if (links[i][j] > 0 and pos[links[i][j]] != -1) {
        links[i][pos[links[i][j]]] *= -1;
        links[i][j] *= -1;
      } else if (links[i][j] > 0 and pos[links[i][j]] == -1)
        pos[links[i][j]] = j;
    }
    fill(all(pos), -1);
  }

  rep(i, 0, SZ(links)) {
    rep(j, 0, SZ(links[i])) {
      cout << links[i][j];
      if (j != SZ(links[i]) - 1) cout << ' ';
    }
    cout << '\n';
  }
  cout << 0 << '\n';  // cont_order count
  return 0;
}
