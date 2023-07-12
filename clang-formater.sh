find . -iname *.hpp -o -iname *.h -o -iname *.cpp -o -iname *.cu | xargs clang-format -i -style=file
