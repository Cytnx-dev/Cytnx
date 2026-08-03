#include "utils/order_parser.hpp"

#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "cytnx_error.hpp"
#include "utils/str_utils.hpp"

namespace cytnx {
  namespace network_internal {
    namespace {

      constexpr std::size_t kMaximumOrderNesting = 1024;

      class OrderParser {
       public:
        OrderParser(const std::string& line, std::size_t line_num)
            : input_line_(line), source_line_num_(line_num) {}

        std::vector<std::string> parse() {
          validate_line_characters();
          skip_whitespace();
          if (cursor_ == input_line_.size()) fail(cursor_, "expected a tensor name or '('");

          parse_tree(0);
          skip_whitespace();

          // The documented examples omit the parentheses around the root, as in
          // "(A,B),(C,D)". Permit exactly one such top-level comma.
          if (cursor_ < input_line_.size() && input_line_[cursor_] == ',') {
            parsed_tokens_.emplace_back(",");
            ++cursor_;
            parse_tree(0);
            skip_whitespace();
          }

          if (cursor_ != input_line_.size()) {
            fail(cursor_, "expected the end of the ORDER expression");
          }
          if (leaf_count_ < 2) {
            fail(cursor_, "expected at least two tensor names");
          }
          return std::move(parsed_tokens_);
        }

       private:
        static bool is_structural(char character) {
          return character == '(' || character == ')' || character == ',';
        }

        void validate_line_characters() const {
          const std::size_t invalid = input_line_.find_first_of("\t;\n:");
          if (invalid != std::string::npos) {
            fail(invalid, "found a character that is not allowed in an ORDER expression");
          }
        }

        void skip_whitespace() {
          while (cursor_ < input_line_.size() &&
                 std::isspace(static_cast<unsigned char>(input_line_[cursor_]))) {
            ++cursor_;
          }
        }

        void parse_tree(std::size_t nesting_depth) {
          skip_whitespace();
          if (cursor_ == input_line_.size()) fail(cursor_, "expected a tensor name or '('");

          if (input_line_[cursor_] != '(') {
            parse_name();
            return;
          }
          if (nesting_depth == kMaximumOrderNesting) {
            fail(cursor_, "ORDER nesting exceeds maximum depth of 1024");
          }

          parsed_tokens_.emplace_back("(");
          ++cursor_;
          parse_tree(nesting_depth + 1);
          consume(',', "expected ',' between the two contraction operands");
          parse_tree(nesting_depth + 1);
          consume(')', "expected ')' after the two contraction operands");
        }

        void parse_name() {
          const std::size_t start = cursor_;
          while (cursor_ < input_line_.size() && !is_structural(input_line_[cursor_])) ++cursor_;

          const std::string name = str_strip(input_line_.substr(start, cursor_ - start));
          if (name.empty()) fail(start, "expected a nonempty tensor name");

          parsed_tokens_.push_back(name);
          ++leaf_count_;
        }

        void consume(char expected, const char* reason) {
          skip_whitespace();
          if (cursor_ == input_line_.size() || input_line_[cursor_] != expected) {
            fail(cursor_, reason);
          }
          parsed_tokens_.emplace_back(1, expected);
          ++cursor_;
        }

        [[noreturn]] void fail(std::size_t error_cursor, const char* reason) const {
          cytnx_error_msg(true,
                          "[ERROR][ORDER] line:%zu column:%zu invalid contraction order: %s\n",
                          source_line_num_, error_cursor + 1, reason);
          std::abort();
        }

        const std::string& input_line_;
        std::size_t source_line_num_;
        std::size_t cursor_ = 0;
        std::size_t leaf_count_ = 0;
        std::vector<std::string> parsed_tokens_;
      };

    }  // namespace

    std::vector<std::string> parse_order_line(const std::string& line, std::size_t line_num) {
      return OrderParser(line, line_num).parse();
    }

  }  // namespace network_internal
}  // namespace cytnx
