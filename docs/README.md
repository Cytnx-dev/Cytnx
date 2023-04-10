# Cytnx_doc
The webpage of Cytnx

# Compilation instructions
1. Install cytnx according to instructions in the documentation:  
    - Install conda  
    - $conda config --add channels conda-forge  
    - $conda create --channel conda-forge --name cytnx python=3.9 _openmp_mutex=*=*_llvm  
    - $conda activate cytnx  
    - $conda install -c kaihsinwu cytnx  

2. Install additional requirements (this might not all be required and/or additional configuration steps might be needed!):  
    - $conda install sphinx sphinxcontrib-bibtex doxygen breathe  
    - $pip install pip install sphinxbootstrap4theme  

3. Run make  
    - $make html  

This will generate the html and the entry point is build/html/index.html
