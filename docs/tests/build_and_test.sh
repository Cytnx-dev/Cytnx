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
rm -r __pycache__
pytest -v test_doc.py
cd "$cur_path"
