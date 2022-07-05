find . -iname *.hpp -o -iname *.h -o -iname *.cpp | xargs clang-format -i -style=file
