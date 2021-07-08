# Cytnx

![alt text](./Icon_small.png)


## Install 
See The following user guide for install and using of cytnx:

[https://kaihsinwu.gitlab.io/Cytnx_doc/install.html](https://kaihsinwu.gitlab.io/Cytnx_doc/install.html)

## Intro slide
[Cytnx_v0.5.pdf (dated 07/25/2020)](https://drive.google.com/file/d/1vuc_fTbwkL5t52glzvJ0nNRLPZxj5en6/view?usp=sharing)


[![Anaconda-Server Badge](https://anaconda.org/kaihsinwu/cytnx/badges/version.svg)](https://anaconda.org/kaihsinwu/cytnx) [![Anaconda-Server Badge](https://anaconda.org/kaihsinwu/cytnx/badges/platforms.svg)](https://anaconda.org/kaihsinwu/cytnx)

## News
    [v0.7.2] 
 
## Stable Version:
[v0.7.2](https://github.com/kaihsin/Cytnx/tree/v0.7.2)

## Known issues:
    v0.7.2
    1. [Pending][GPU] Get/Set elements on GPU is still down. 


## Current dev Version:
    v0.7.3
    1. [Fix] bug for Get slice does not reduce when dim=1. 
    2. [Enhance] checking the memory alloc failing for EL.  
    3. [Change] remove Tensor init assignment op from initializer_list, for conflict with UniTensor init.    
    4. [Enhance] print information for Symmetric UniTensor.
    5. [Enhance] linalg::ExpM/ExpH support for symmetric UniTensor.
    6. [Enhance] add UniTensor.get_blocks_qnums() for corresponding qnums for current blocks. 
    7. [Enhance][Safety] add UniTensor.get_blocks_(silent=false) with "silent" option by default pop-up a warning when UniTensor is non-contiguous.   
    8. [Enhance] add operator* and operator*= for combineBond. 
    9. [Enhance] add support for Symmetric UniTensor with is_diag=true.
    10. [Fix] remove the dtype & device option for arange(Nelem). Use .astype() .to() instead. 
    11. [Fix] reshape() without postfix const causing error when reshape with const Tensor. 
    12. [Enhance][Experiment] add Lstsq for least square calculation. [PR] 
    13. [Fix][C++] minor issue related to laterial argument passing by variables cannot properly resolved on C++ 
    14. [Enhance] Diag now support rank-1 Tensor as input for constructing a diagonal tensor with input as diagonal elements.
    15. [Enhance] Add c++ example for DMRG (Ke)
    16. [Fix] Bug fixed in DMRG code and updated to the latest features. 
    17. [Fix] Bug in UniTensor do svd with rowrank=1 and the first rank has dimension=1.        

    v0.7.2 
    1. [Enhance] Add Tensor.set with Scalar
    2. [Enhance][C++] Add Tensor initialize assignment op from initializer_list
    3. [Enhance][C++] Add Storage initialize assignment op from vector & initializer list  
    4. [Fix] bug for set partial elements on Tensor with slicing issue. 
    5. [Fix][DenseUniTensor] set_rowrank cannot set full rank issue #24 


    v0.7.1
    1. [Enhance] Finish UniTensor arithmetic. 
    2. [Fix] bug when using Tensor.get() accessing only single element 
    3. [Enhance] Add default argument is_U = True and is_vT = True for Svd_truncate() python API 


    v0.7
    1. [Enhance] add binary op. -Tensor.    
    2. [Enhance] New introduce Scalar class, generic scalar placeholder. 
    3. [Enhance][expr] Storage.at(), Storage.back(), Storage.get_item() can now without specialization. The return is Scalar class.
    4. [Enhance] Storage.get_item, Storage.set_item        
    5. [Enhance] Scalar, iadd,isub,imul,idiv  
    6. [Important] Storage.resize will match the behavior of vector, new elements are set to zero!
    7. [Enhance] Scalar +,-,*,/ finished
    8. [Enhance] add Histogram class and stat namespace.    
    9. [Enhance] add fstream option for Tofile
    10. [Enhance] return self when UniTensor.set_name
    11. [Enhance] return self when UniTensor.set_label(s)
    12. [Enhance] return self when UniTensor.set_rowrank
    13. [Fatal!][Fix] fix bug of wrong answer in Tensor slice for non-contiguous Tensor, with faster internal kernel
    14. [Warning] Slice of GPU Tensor is now off-line for further inspection. 
    15. [Fix] bug causing crash when print non-contiguous Uint64 Tensor    
    16. [Fatal!][Fix] fix bug of wrong answer in Tensor set-element with slice for non-contiguous Tensor. 
    17. [Enhance] Network on the fly construction.
    18. [Enhance] Scalar: Add on TN. TN.item()
    19. [Fix] bug in Mod interanlly calling Cpr fixed.    
    20. [Enhance] All operation related to TN <-> Scalar
    21. [Enhance] Reduce RTTR overhead. 
 

    v0.6.5
    1. [Fix] Bug in UniTensor _Load    
    2. [Enhance] Improve stability in Lanczos_ER  
    3. [Enhance] Move _SII to stack.
    4. [Enhance] Add LinOp operator() for mv_elem
    5. [Enhance] Add c++ API fast link to cutt
    6. [Enhance] Add Fromfile/Tofile for load/save binary files @ Tensor/Storage
    7. [Enhance] Add linspace generator
    8. [Fix] Bug in Div for fast Blas call bug
    9. [Enhance] Add Tensor.append(Storage) if Tensor is rank-2 and dimension match.
    10. [Enhance] Add algo namespace
    11. [Enhance] Add Sort-@cpu
    12. [Enhance] add Storage.numpy() for pythonAPI
    13. [Enhance] add Tensor.from_storage() for python API
    
    v0.6.4
    1. [Enhance] Add option mv_elem for Tensordot, which actually move elements in input tensor. This is beneficial when same tensordot is called multiple times.
    2. [Enhance] Add option cacheL, cacheR to Contract of unitensor. which mv the elements of input tensors to the matmul handy position. 
    3. [Enhance] optimize Network contraction policy to reduce contiguous permute, with is_clone argument when PutUniTensor.
    4. [Enhance] Add Lanczos_Gnd for fast get ground state and it's eigen value (currently only real float). 
    5. [Enhance] Add Tridiag python API, and option is_row
    6. [Enhance] C++ API storage add .back<>() function. 
    7. [Enhance] C++ API storage fix from_vector() for bool type. 
    8. [Enhance] Change Network Launch optimal=True behavior. if user order is given, optimal will not have effect.   
    9. [Enhance] Add example/iDMRG/dmrg_optim.py for better performace with Lanczos_Gnd and Network cache.
    10. [Fix] wrong error message in linalg::Cpr
    11. [Fix] reshape() on a already contiguous Tensor will resulting as the change in original tensor, which should not happened.



## API Documentation:

[https://kaihsin.github.io/Cytnx/docs/html/index.html](https://kaihsin.github.io/Cytnx/docs/html/index.html)

## User Guide [under construction!]:

[Cytnx User Guide](https://kaihsinwu.gitlab.io/Cytnx_doc/)



## Objects:
    * Storage   [binded]
    * Tensor    [binded]
    * Accessor  [c++ only]
    * Bond      [binded] 
    * Symmetry  [binded] 
    * CyTensor [binded] 
    * Network   [binded] 

## Feature:

### Python x C++
    Benefit from both side. 
    One can do simple prototype on python side 
    and easy transfer to C++ with small effort!


```c++
    // c++ version:
    #include "cytnx.hpp"
    cytnx::Tensor A({3,4,5},cytnx::Type.Double,cytnx::Device.cpu)
```


```python
    # python version:
    import cytnx
    A =  cytnx.Tensor((3,4,5),dtype=cytnx.Type.Double,device=cytnx.Device.cpu)
```


### 1. All the Storage and Tensor can now have mulitple type support. 
        The avaliable types are :

        | cytnx type       | c++ type             | Type object
        |------------------|----------------------|--------------------
        | cytnx_double     | double               | Type.Double
        | cytnx_float      | float                | Type.Float
        | cytnx_uint64     | uint64_t             | Type.Uint64
        | cytnx_uint32     | uint32_t             | Type.Uint32
        | cytnx_uint16     | uint16_t             | Type.Uint16
        | cytnx_int64      | int64_t              | Type.Int64
        | cytnx_int32      | int32_t              | Type.Int32
        | cytnx_int16      | int16_t              | Type.Int16
        | cytnx_complex128 | std::complex<double> | Type.ComplexDouble
        | cytnx_complex64  | std::complex<float>  | Type.ComplexFloat
        | cytnx_bool       | bool                 | Type.Bool

### 2. Storage
        * Memory container with GPU/CPU support. 
          maintain type conversions (type casting btwn Storages) 
          and moving btwn devices.
        * Generic type object, the behavior is very similar to python.

```c++
            Storage A(400,Type.Double);
            for(int i=0;i<400;i++)
                A.at<double>(i) = i;

            Storage B = A; // A and B share same memory, this is similar as python 
            
            Storage C = A.to(Device.cuda+0); 
```


### 3. Tensor
        * A tensor, API very similar to numpy and pytorch.
        * simple moving btwn CPU and GPU:

```c++
            Tensor A({3,4},Type.Double,Device.cpu); // create tensor on CPU (default)
            Tensor B({3,4},Type.Double,Device.cuda+0); // create tensor on GPU with gpu-id=0


            Tensor C = B; // C and B share same memory.

            // move A to gpu
            Tensor D = A.to(Device.cuda+0);

            // inplace move A to gpu
            A.to_(Device.cuda+0);
```
        * Type conversion in between avaliable:
```c++
            Tensor A({3,4},Type.Double);
            Tensor B = A.astype(Type.Uint64); // cast double to uint64_t
```

        * vitual swap and permute. All the permute and swap will not change the underlying memory
        * Use Contiguous() when needed to actual moving the memory layout.
```c++
            Tensor A({3,4,5,2},Type.Double);
            A.permute_(0,3,1,2); // this will not change the memory, only the shape info is changed.
            cout << A.is_contiguous() << endl; // this will be false!

            A.contiguous_(); // call Configuous() to actually move the memory.
            cout << A.is_contiguous() << endl; // this will be true!
```

        * access single element using .at
```c++
            Tensor A({3,4,5},Type.Double);
            double val = A.at<double>(0,2,2);
```

        * access elements with python slices similarity:
```c++
            typedef Accessor ac;
            Tensor A({3,4,5},Type.Double);
            Tensor out = A(0,":","1:4"); 
            // equivalent to python: out = A[0,:,1:4]
            
```

### 4. UniTensor
        * extension of Tensor, specifically design for Tensor network simulation. 

        * See Intro slide for more details
```c++
            Tensor A({3,4,5},Type.Double);
            UniTensor tA = UniTensor(A,2); // convert directly.

            UniTensor tB = UniTensor({Bond(3),Bond(4),Bond(5)},{},2); // init from scratch. 
```




## Examples
    
    See example/ folder or documentation for how to use API
    See example/iTEBD folder for implementation on iTEBD algo.
    See example/DMRG folder for implementation on DMRG algo.
    See example/iDMRG folder for implementation on iDMRG algo.
    See example/HOTRG folder for implementation on HOTRG algo for classical system.
    See example/ED folder for implementation using LinOp & Lanczos. 


## Avaliable linear-algebra function (Keep updating):

      func        |   inplace | CPU | GPU  | callby tn   | Tn | CyTn (xlinalg)
    --------------|-----------|-----|------|-------------|----|-------
      Add         |   x       |  Y  |  Y   |    Y        | Y  |   Y
      Sub         |   x       |  Y  |  Y   |    Y        | Y  |   Y
      Mul         |   x       |  Y  |  Y   |    Y        | Y  |   Y
      Div         |   x       |  Y  |  Y   |    Y        | Y  |   Y
      Cpr         |   x       |  Y  |  Y   |    Y        | Y  |   x
    --------------|-----------|-----|------|-------------|----|-------
      +,+=[tn]    |   x       |  Y  |  Y   |    Y (Add_) | Y  |   Y
      -,-=[tn]    |   x       |  Y  |  Y   |    Y (Sub_) | Y  |   Y
      *,*=[tn]    |   x       |  Y  |  Y   |    Y (Mul_) | Y  |   Y
      /,/=[tn]    |   x       |  Y  |  Y   |    Y (Div_) | Y  |   Y
      ==[tn]      |   x       |  Y  |  Y   |    Y (Cpr_) | Y  |   x 
    --------------|-----------|-----|------|-------------|----|-------
      Svd         |   x       |  Y  |  Y   |    Y        | Y  |   Y
     *Svd_truncate|   x       |  Y  |  Y   |    N        | Y  |   Y
      InvM        |   InvM_   |  Y  |  Y   |    Y        | Y  |   N
      Inv         |   Inv _   |  Y  |  Y   |    Y        | Y  |   N
      Conj        |   Conj_   |  Y  |  Y   |    Y        | Y  |   Y
    --------------|-----------|-----|------|-------------|----|-------
      Exp         |   Exp_    |  Y  |  Y   |    Y        | Y  |   N
      Expf        |   Expf_   |  Y  |  Y   |    Y        | Y  |   N
      Eigh        |   x       |  Y  |  Y   |    Y        | Y  |   N
     *ExpH        |   x       |  Y  |  Y   |    N        | Y  |   Y
     *ExpM        |   x       |  Y  |  N   |    N        | Y  |   Y
    --------------|-----------|-----|------|-------------|----|-------
      Matmul      |   x       |  Y  |  Y   |    N        | Y  |   N
      Diag        |   x       |  Y  |  Y   |    N        | Y  |   N
    *Tensordot    |   x       |  Y  |  Y   |    N        | Y  |   N
     Outer        |   x       |  Y  |  Y   |    N        | Y  |   N 
     Vectordot    |   x       |  Y  | .Y   |    N        | Y  |   N 
    --------------|-----------|-----|------|-------------|----|-------
      Tridiag     |   x       |  Y  |  N   |    N        | Y  |   N
     Kron         |   x       |  Y  |  N   |    N        | Y  |   N
     Norm         |   x       |  Y  |  Y   |    Y        | Y  |   N
    *Dot          |   x       |  Y  |  Y   |    N        | Y  |   N 
     Eig          |   x       |  Y  |  N   |    N        | Y  |   N 
    --------------|-----------|-----|------|-------------|----|-------
     Pow          |   Pow_    |  Y  |  Y   |    Y        | Y  |   Y 
     Abs          |   Abs_    |  Y  |  N   |    Y        | Y  |   N 
     Qr           |   x       |  Y  |  N   |    N        | Y  |   Y 
     Qdr          |   x       |  Y  |  N   |    N        | Y  |   Y 
     Det          |   x       |  Y  |  N   |    N        | Y  |   N
    --------------|-----------|-----|------|-------------|----|-------
     Min          |   x       |  Y  |  N   |    Y        | Y  |   N 
     Max          |   x       |  Y  |  N   |    Y        | Y  |   N 
    *Trace        |   x       |  Y  |  N   |    Y        | Y  |   Y
     Mod          |   x       |  Y  |  Y   |    Y        | Y  |   Y 
    Matmul_dg     |   x       |  Y  |  Y   |    N        | Y  |   N 
    --------------|-----------|-----|------|-------------|----|-------
    *Tensordot_dg |   x       |  Y  |  Y   |    N        | Y  |   N

    iterative solver:
     
        Lanczos_ER           
    

    * this is a high level linalg 
    
    ^ this is temporary disable
    
    . this is floating point type only
 
## Container Generators 

    Tensor: zeros(), ones(), arange(), identity(), eye()

## Physics category 

    Tensor: pauli(), spin()
    
     
## Random 
      func        | Tn  | Stor | CPU | GPU  
    -----------------------------------------------------
    *Make_normal() |  Y  |  Y   | Y   |  Y
    *Make_uniform() |  Y  |  Y   | Y   |  N
    ^normal()      |  Y  |  x   | Y   |  Y
    ^uniform()      |  Y  |  x   | Y   |  N

    * this is initializer
    ^ this is generator

    [Note] The difference of initializer and generator is that initializer is used to initialize the Tensor, and generator generates a new Tensor.
     

## Developer

    Kai-Hsin Wu (Boston Univ.) kaihsinwu@gmail.com 


## Contributors

    Ying-Jer Kao (NTU, Taiwan): setuptool, cmake
    Yen-Hsin Wu (NTU, Taiwan): Network optimization
    Yu-Hsueh Chen (NTU, Taiwan): example, and testing
    Po-Kwan Wu (OSU): Icon optimization    
    Wen-Han Kao (UMN, USA) : testing of conda install 
    Ke Hsu (NTU, Taiwan): Lstsq, linalg funcitons and examples  

## References

    * example/DMRG:
        https://www.tensors.net/dmrg

    * hptt library:
        https://github.com/springer13/hptt


## Acknowledgement
    KHW whould like to thanks for the following contributor(s) for invaluable contribution to the library

    * PoChung Chen  (NCHU, Taiwan) : testing, and bug reporting


