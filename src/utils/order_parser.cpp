#include "utils/order_parser.hpp"

#include <cctype>
#include <cstdlib>
#include <string>
#include <vector>

#include "cytnx_error.hpp"
#include "utils/str_utils.hpp"

namespace cytnx {
  namespace network_internal {
    namespace {

      class OrderParser {
       public:
        OrderParser(const std::string& line, std::size_t line_num)
            : input_line(line), source_line_num(line_num) {}

        std::vector<std::string> parse() {
          validate_line_characters();
          skip_whitespace();
          if (cursor == input_line.size()) fail(cursor, "expected a tensor name or '('");

          parse_tree();
          skip_whitespace();

          // The documented examples omit the parentheses around the root, as in
          // "(A,B),(C,D)". Permit exactly one such top-level comma.
          if (cursor < input_line.size() && input_line[cursor] == ',') {
            parsed_tokens.emplace_back(",");
            ++cursor;
            parse_tree();
            skip_whitespace();
          }

          if (cursor != input_line.size()) {
            fail(cursor, "expected the end of the ORDER expression");
          }
          if (leaf_count < 2) {
            fail(cursor, "expected at least two tensor names");
          }
          return parsed_tokens;
        }

       private:
        const std::string& input_line;
        std::size_t source_line_num;
        std::size_t cursor = 0;
        std::size_t leaf_count = 0;
        std::vector<std::string> parsed_tokens;

        static bool is_structural(char character) {
          return character == '(' || character == ')' || character == ',';
        }

        void validate_line_characters() const {
          const std::size_t invalid = input_line.find_first_of("\t;\n:");
          if (invalid != std::string::npos) {
            fail(invalid, "found a character that is not allowed in an ORDER expression");
          }
        }

        void skip_whitespace() {
          while (cursor < input_line.size() &&
                 std::isspace(static_cast<unsigned char>(input_line[cursor]))) {
            ++cursor;
          }
        }

        void parse_tree() {
          skip_whitespace();
          if (cursor == input_line.size()) fail(cursor, "expected a tensor name or '('");

          if (input_line[cursor] != '(') {
            parse_name();
            return;
          }

          parsed_tokens.emplace_back("(");
          ++cursor;
          parse_tree();
          consume(',', "expected ',' between the two contraction operands");
          parse_tree();
          consume(')', "expected ')' after the two contraction operands");
        }

        void parse_name() {
          const std::size_t start = cursor;
          while (cursor < input_line.size() && !is_structural(input_line[cursor])) ++cursor;

          const std::string name = str_strip(input_line.substr(start, cursor - start));
          if (name.empty()) fail(start, "expected a nonempty tensor name");

          parsed_tokens.push_back(name);
          ++leaf_count;
        }

        void consume(char expected, const char* reason) {
          skip_whitespace();
          if (cursor == input_line.size() || input_line[cursor] != expected) fail(cursor, reason);
          parsed_tokens.emplace_back(1, expected);
          ++cursor;
        }

        [[noreturn]] void fail(std::size_t error_cursor, const char* reason) const {
          cytnx_error_msg(true,
                          "[ERROR][ORDER] line:%zu column:%zu invalid contraction order: %s\n",
                          source_line_num, error_cursor + 1, reason);
          std::abort();
        }
      };

    }  // namespace

    std::vector<std::string> parse_order_line(const std::string& line, std::size_t line_num) {
      return OrderParser(line, line_num).parse();
    }

  }  // namespace network_internal
}  // namespace cytnx
