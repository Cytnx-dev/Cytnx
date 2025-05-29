# Cytnx_doc unit test
This is about the documentation unit test. The documentation unit test is a test that runs the code in the documentation.

# How to run the unit test
1. If you want to run the unit test in python version, you need to install 'pytest' by using the following command:
```bash
$ conda install -c anaconda pytest
```
2. Modify the file 'CMakelists.txt' to install Cytnx correctly.
3. Run the following command to run the unit test:
```bash
$ sh build_and_test.sh
```
This command will build the Cytnx and run the unit test for both C++ and python version. The excutable test file is in the following path:
+ C++: ./build/test_doc_cplusplus
+ python: ./test_doc.py
