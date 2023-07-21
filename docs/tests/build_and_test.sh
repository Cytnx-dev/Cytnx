cur_path=$(pwd)

# build c++ test

#rm -rf build
mkdir build
cd build
cmake ..
make
#run c++ test
./test_doc_cplusplus

#run python test
cd "$cur_path"
pytest -v test_doc.py
