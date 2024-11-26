#ifndef _H_search_tree_
#define _H_search_tree_

#include "Type.hpp"
#include "cytnx_error.hpp"
#include "utils/utils.hpp"
#include "Type.hpp"
#include <vector>
#include <map>
#include <string>
#include <memory>
#include "UniTensor.hpp"

#ifdef BACKEND_TORCH
#else

namespace cytnx {
  class objectData {
   public:
    objectData();
    ~objectData() = default;

    std::vector<bool> legLinks;
    std::vector<bool> tensorflags;  // Length is always numtensors
    cytnx_size_t sequencelen;
    std::vector<cytnx_int64> sequence;
    cytnx_int64 costlen;
    std::vector<double> cost;
    bool isOP;
    std::vector<double> maxdim;  // Length 2 if costType==2, length 1 if costType==1
    std::vector<double> allIn;
    bool isnew;
  };

  objectData::objectData() : sequencelen(0), costlen(0), isOP(false), isnew(false) {}

  class objectList {
   public:
    objectList();
    ~objectList() = default;

    std::shared_ptr<objectList> next;
    std::shared_ptr<objectData> object;
  };
  objectList::objectList() : next(nullptr), object(nullptr) {}

  class objectTree {
   public:
    objectTree(const cytnx_int64 numtensors, const cytnx_int64 offset);
    ~objectTree() = default;
    inline std::shared_ptr<objectTree> getbranch(const cytnx_int64 branch);
    inline void setbranch(const cytnx_int64 branch, std::shared_ptr<objectTree> target);

    cytnx_int64 numentries;
    std::shared_ptr<objectData> object;
    cytnx_int64 offset;

   private:
    std::vector<std::shared_ptr<objectTree>> branches;
  };

  objectTree::objectTree(const cytnx_int64 numtensors, const cytnx_int64 o)
      : numentries(numtensors - o), offset(o), branches(numentries, nullptr), object(nullptr) {}

  class tensorXlist {
   public:
    tensorXlist();
    ~tensorXlist() = default;

    std::shared_ptr<tensorXlist> next;
    std::shared_ptr<tensorXlist> prev;
    std::vector<bool> legs;
    std::vector<double> allIn;  // For a tensor whose construction is consistent with Fig.5(c),
                                // allIn is the dimension of index c, corresponding to |E|.
    cytnx_int64 flag;
  };

  tensorXlist::tensorXlist() : next(nullptr), prev(nullptr), flag(2) {}

}  // namespace cytnx

#endif
#endif
