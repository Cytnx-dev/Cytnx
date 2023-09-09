var=$(find . -iname *.hpp -o -iname *.h -o -iname *.cpp -o -iname *.cu)
echo $var
clang-format -i $var -style=file
