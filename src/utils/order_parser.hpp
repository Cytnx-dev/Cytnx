#ifndef CYTNX_UTILS_ORDER_PARSER_H_
#define CYTNX_UTILS_ORDER_PARSER_H_

#include <cstddef>
#include <string>
#include <vector>

namespace cytnx {
  namespace network_internal {

    // Parse a binary contraction order. A tree is either a tensor name or
    // "(" tree "," tree ")"; the outermost parentheses may be omitted.
    std::vector<std::string> parse_order_line(const std::string& line, std::size_t line_num);

  }  // namespace network_internal
}  // namespace cytnx

#endif  // CYTNX_UTILS_ORDER_PARSER_H_
