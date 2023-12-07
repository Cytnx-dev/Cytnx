# Cytnx_doc
The documentation of the Cytnx tensor network library


# Compilation instructions
1. Install cytnx according to instructions in the documentation:  
    Please install conda first.
    ```bash
    - $ conda config --add channels conda-forge  
    - $ conda create --channel conda-forge --name cytnx python=3.9 _openmp_mutex=*=*_llvm  
    - $ conda activate cytnx  
    - $ conda install -c kaihsinwu cytnx  
    ```

2. Install additional requirements (this might not all be required and/or additional configuration steps might be needed!):
    ```bash
    - $ conda install sphinx sphinxcontrib-bibtex doxygen breathe sphinxcontrib-jquery 
    - $ pip install sphinxbootstrap4theme  
    ```

3. Run make  
    ```bash
    $ make html  
    ```

     This will generate the html and the entry point is build/html/index.html
4. If you want to run the unit test in the documentation, see [here](./tests/README.md).
