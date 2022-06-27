#include "Accessor.hpp"
#include "utils/str_utils.hpp"
#include <iostream>
#include <algorithm>
#include <utility>
#include "utils/vec_print.hpp"
using namespace std;
namespace cytnx {

  Accessor::Accessor(const cytnx_int64 &loc) {
    this->_type = Accessor::Singl;
    this->loc = loc;
  }

  /*
  Accessor::Accessor(const Tensor &tn){
      this->_type = Accessor::Tn;
      if(Type.is_int(tn.dtype()){
          cytnx_error_msg(tn.shape().size()!=1,"[ERROR] Accessor with Tensor can only accept
  rank=1.%s","\n"); cytnx_error_msg(tn.device()!=Device.cpu,"[ERROR] Accessor with Tensor can only
  accept Tensor on cpu.\n Suggestion: Use to_() or to() first.%s","\n"); this->idx_list.clear();
          if(tn.dtype()!=Type.Int64){
              Tensor tmp = tn.astype(Type.Int64);
              this->idx_list.resize(tmp.storage().size());
              memcpy(&(this->idx_list[0]),tmp.storage().data<cytnx_int64>(),sizeof(cytnx_int64)*this->idx_list.size());
          }else{
              this->idx_list.resize(tn.storage().size());
              memcpy(&(this->idx_list[0]),tn.storage().data<cytnx_int64>(),sizeof(cytnx_int64)*this->idx_list.size());
          }

      }else{
          cytnx_error_msg(true,"[ERROR] Accessor with Tensor can only accept int type.%s","\n");
      }

  }
  */

  // all constr. ( use string to dispatch )
  Accessor::Accessor(const std::string &str) {
    // this->_axis_len = 0;

    // std::cout << str << "|" << std::endl;
    if ((str == "all") || (str == ":"))
      this->_type = Accessor::All;
    else {
      // cytnx_error_msg(true,"[ERROR] only Accessor::all() can use string to init.%s","\n");
      std::vector<std::string> token = str_split(str, false, ":");

      cytnx_error_msg(token.size() <= 1,
                      "[ERROR] no ':' in resolving accessor. use integer directly.%s", "\n");
      cytnx_error_msg(token.size() > 3,
                      "[ERROR] invalid string to Accessor, make sure no space exist.%s", "\n");

      /// min :
      this->_min = token[0].size() == 0 ? 0 : std::stoll(token[0]);

      /// max :
      if (token[1].size() == 0) {
        this->_type = Accessor::Tilend;
      } else {
        this->_type = Accessor::Range;
        this->_max = std::stoll(token[1]);
      }

      /// step:
      if (token.size() == 3) {
        if (token[2].size() == 0) {
          this->_step = 1;
        } else {
          this->_step = std::stoll(token[2]);
        }
        if ((token[0].size() == 0) && (token[1].size() == 0)) {
          this->_type = Accessor::Step;
        }
      } else {
        this->_step = 1;
      }

      // cout << this->min;
      // cout << this->max;
      // cout << this->step << endl;
      // std::cout << token << std::endl;
    }
  }

  // range constr.
  Accessor::Accessor(const cytnx_int64 &min, const cytnx_int64 &max, const cytnx_int64 &step) {
    cytnx_error_msg(step == 0, "[ERROR] cannot have step=0 for range%s", "\n");
    this->_type = Accessor::Range;
    this->_min = min;
    this->_max = max;
    this->_step = step;
  }

  // copy constructor:
  Accessor::Accessor(const Accessor &rhs) {
    this->_type = rhs._type;
    this->_min = rhs._min;
    this->_max = rhs._max;
    this->loc = rhs.loc;
    this->_step = rhs._step;
    this->idx_list = rhs.idx_list;
  }

  // copy assignment:
  Accessor &Accessor::operator=(const Accessor &rhs) {
    this->_type = rhs._type;
    this->_min = rhs._min;
    this->_max = rhs._max;
    this->loc = rhs.loc;
    this->_step = rhs._step;
    this->idx_list = rhs.idx_list;
    return *this;
  }

  // get the real len from dim
  // if _type is all, pos will be null, and len == dim
  // if _type is range, pos will be the locator, and len == len(pos)
  // if _type is singl, pos will be pos, and len == 0
  void Accessor::get_len_pos(const cytnx_uint64 &dim, cytnx_uint64 &len,
                             std::vector<cytnx_uint64> &pos) const {
#ifdef UNI_DEBUG
    cytnx_error_msg(this->_type == Accessor::none, "%s",
                    "[DEBUG][ERROR] try to call get_len from an un-initialize Accessor.");
#endif

    pos.clear();

    if (this->_type == Accessor::All) {
      len = dim;
    } else if (this->_type == Accessor::Range) {
      cytnx_int64 r_min = this->_min < 0 ? (this->_min + dim) % dim : this->_min;
      cytnx_int64 r_max = this->_max < 0 ? (this->_max + dim) % dim : this->_max;
      cytnx_error_msg((r_min < 0 || r_max < 0), "%s", "[ERROR] index is out of bounds\n");
      cytnx_error_msg((r_max - r_min) / this->_step <= 0, "%s",
                      "[ERROR] upper bound and larger bound inconsistent with step sign resulting "
                      "a null Tensor.");

      // len = (r_max-r_min)/this->step;
      // std::cout << len << " " << dim << std::endl;
      // if((r_max-r_min)%this->step) len+=1;

      len = 0;
      if (this->_step < 0) {
        for (cytnx_int64 i = r_min; i > r_max; i += this->_step) {
          pos.push_back(i);
          // std::cout << pos.back() << std::endl;
          len++;
        }
      } else {
        for (cytnx_int64 i = r_min; i < r_max; i += this->_step) {
          pos.push_back(i);
          // std::cout << pos.back() << std::endl;
          len++;
        }
      }

    } else if (this->_type == Accessor::Tilend) {
      cytnx_int64 r_min = this->_min, r_max = dim;
      if (this->_step < 0) {
        r_min = (this->_min + dim) % dim;
        r_max = 0;
      } else {
        r_min = (this->_min + dim) % dim;
        r_max = dim - 1;
      }
      cytnx_error_msg((r_min < 0 || r_max < 0), "%s", "[ERROR] index is out of bounds\n");
      cytnx_error_msg((r_max - r_min) / this->_step < 0, "%s",
                      "[ERROR] upper bound and larger bound inconsistent with step sign");
      len = 0;
      if (this->_step < 0) {
        for (cytnx_int64 i = r_min; i >= r_max; i += this->_step) {
          pos.push_back(i);
          // std::cout << pos.back() << std::endl;
          len++;
        }
      } else {
        for (cytnx_int64 i = r_min; i <= r_max; i += this->_step) {
          pos.push_back(i);
          // std::cout << pos.back() << std::endl;
          len++;
        }
      }

    } else if (this->_type == Accessor::Singl) {
      // check:
      // std::cout << this->loc << " " << dim << std::endl;
      cytnx_error_msg(std::abs(this->loc) >= dim, "[ERROR] index is out of bound%s", "\n");
      len = 1;
      if (this->loc < 0)
        pos.push_back(this->loc + dim);
      else
        pos.push_back(this->loc);
    } else if (this->_type == Accessor::Step) {
      cytnx_int64 r_min, r_max;
      if (this->_step < 0) {
        r_min = dim - 1;
        r_max = 0;
      } else {
        r_min = 0;
        r_max = dim - 1;
      }
      cytnx_error_msg((r_max - r_min) / this->_step <= 0, "%s",
                      "[ERROR] upper bound and larger bound inconsistent with step sign");

      len = 0;
      if (this->_step < 0) {
        for (cytnx_int64 i = r_min; i >= r_max; i += this->_step) {
          pos.push_back(i);
          // std::cout << pos.back() << std::endl;
          len++;
        }
      } else {
        for (cytnx_int64 i = r_min; i <= r_max; i += this->_step) {
          pos.push_back(i);
          // std::cout << pos.back() << std::endl;
          len++;
        }
      }

    } else if (this->_type == Accessor::list) {
      pos.clear();
      pos.resize(this->idx_list.size());
      len = pos.size();
// cout << "list in accessor len:" <<len << endl;
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 i = 0; i < this->idx_list.size(); i++) {
        // checking:
        if (this->idx_list[i] < 0) {
          cytnx_error_msg(this->idx_list[i] + dim < 0,
                          "[ERROR] invalid position at %d in accessor. type: list.\n", i);
          pos[i] = this->idx_list[i] + dim;
        } else {
          cytnx_error_msg(this->idx_list[i] >= dim,
                          "[ERROR] invalid position at %d in accessor. type: list.\n", i);
          pos[i] = idx_list[i];
        }
      }
    }
  }
  //============================================
  std::ostream &operator<<(std::ostream &os, const Accessor &in) {
    if (in.type() == Accessor::Singl) {
      os << in.loc;
    } else if (in.type() == Accessor::All) {
      os << ":";
    } else if (in.type() == Accessor::Range) {
      os << in._min << ":" << in._max << ":" << in._step;
    } else if (in.type() == Accessor::Tilend) {
      if (in._step == 1)
        os << in._min << ":";
      else
        os << in._min << ":"
           << ":" << in._step;
    } else if (in.type() == Accessor::Step) {
      os << "::" << in._step;
    } else if (in.type() == Accessor::Qns) {
      os << "Qnum select: " << in.qns_list.size() << " qnums:" << endl;
      for (int i = 0; i < in.qns_list.size(); i++) {
        os << " {";
        for (int j = 0; j < in.qns_list[i].size(); j++) {
          if (j == 0)
            os << in.qns_list[i][j];
          else
            os << "," << in.qns_list[i][j];
        }
        os << "}";
      }
    } else {
      cytnx_error_msg(true, "[ERROR][cout] Accessor is Void!%s", "\n");
    }
    return os;
  }

}  // namespace cytnx
