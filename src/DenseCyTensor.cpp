#include "CyTensor.hpp"
#include "utils/utils.hpp"
#include "utils/utils_internal_interface.hpp"
#include "Generator.hpp"
#include "linalg.hpp"
#include <algorithm>
#include <utility>
#include <vector>
namespace cytnx_extension{
    using namespace cytnx;

    void DenseCyTensor::Init(const std::vector<Bond> &bonds, const std::vector<cytnx_int64> &in_labels, const cytnx_int64 &Rowrank, const unsigned int &dtype,const int &device, const bool &is_diag){

                //check for all bonds
                this->_is_tag =false;
                cytnx_uint32 N_ket = 0;
                if(bonds.size()!=0) this->_is_tag = (bonds[0].type() != bondType::BD_REG);
                for(cytnx_uint64 i=0;i<bonds.size();i++){
                    //check 
                    cytnx_error_msg(bonds[i].qnums().size()!=0,"%s","[ERROR][DenseCyTensor] All bonds must have non symmetries.");
                    if(this->_is_tag){
                        cytnx_error_msg(bonds[i].type() == bondType::BD_REG,"%s","[ERROR][DenseCyTensor] cannot mix tagged bond with un-tagged bond!%s","\n");
                        N_ket += cytnx_uint32(bonds[i].type() == bondType::BD_KET);
                    }else{
                        cytnx_error_msg(bonds[i].type() != bondType::BD_REG,"%s","[ERROR][DenseCyTensor] cannot mix tagged bond with un-tagged bond!%s","\n");
                    }
                    cytnx_error_msg(bonds[i].dim()==0,"%s","[ERROR] All bonds must have dimension >=1");
                }
                
                //check Rowrank
                if(this->_is_tag){
                    if(Rowrank < 0){this->_Rowrank = N_ket;}
                    else{
                        cytnx_error_msg(Rowrank > bonds.size(),"[ERROR] Rowrank cannot exceed total rank of Tensor.%s","\n");
                        this->_Rowrank = Rowrank;
                    }
                }else{ 
                    if(bonds.size()==0) this->_Rowrank = 0;    
                    else{
                        cytnx_error_msg(Rowrank <0, "[ERROR] initialize a non-symmetry, un-tagged tensor should assign a >=0 Rowrank.%s","\n");
                        cytnx_error_msg(Rowrank > bonds.size(),"[ERROR] Rowrank cannot exceed total rank of Tensor.%s","\n");
                        this->_Rowrank = Rowrank;
                    }
                }
                                
                //check labels:
                if(in_labels.size()==0){
                    for(cytnx_int64 i=0;i<bonds.size();i++)
                        this->_labels.push_back(i);
    
                }else{
                    //check bonds & labels dim                 
                    cytnx_error_msg(bonds.size()!=in_labels.size(),"%s","[ERROR] labels must have same lenth as # of bonds.");

                    std::vector<cytnx_int64> tmp = vec_unique(in_labels);
                    cytnx_error_msg(tmp.size()!=in_labels.size(),"[ERROR] labels cannot contain duplicated elements.%s","\n");
                    this->_labels = in_labels;
                }

                if(is_diag){
                    cytnx_error_msg(bonds.size()!=2,"[ERROR] is_diag= ture should have the shape for initializing the CyTensor is square, 2-rank tensor.%s","\n");
                    cytnx_error_msg(bonds[0].dim() != bonds[1].dim(),"[ERROR] is_diag= ture should have the shape for initializing the CyTensor is square, 2-rank tensor.%s","\n");
                }


                //copy bonds, otherwise it will share objects:
                this->_bonds = vec_clone(bonds);
                this->_is_braket_form = this->_update_braket();


                //non symmetry, initialize memory.
                if(this->_bonds.size()==0){
                    //scalar:
                    this->_block = zeros({1},dtype,device);
                }else{
                    if(is_diag){
                        this->_block = zeros({_bonds[0].dim()},dtype,device);
                        this->_is_diag = is_diag;
                    }else{
                        std::vector<cytnx_uint64> _shape(bonds.size());
                        for(unsigned int i=0;i<_shape.size();i++)
                            _shape[i] = bonds[i].dim();

                        this->_block = zeros(_shape,dtype,device);
                    }
                }          
    }

    void DenseCyTensor::Init_by_Tensor(const Tensor& in_tensor, const cytnx_uint64 &Rowrank){
                cytnx_error_msg(in_tensor.dtype() == Type.Void,"[ERROR][Init_by_Tensor] cannot init a CyTensor from an un-initialize Tensor.%s","\n");
                if(in_tensor.storage().size() == 1){
                    //scalalr:
                    cytnx_error_msg(Rowrank != 0, "[ERROR][Init_by_Tensor] detect the input Tensor is a scalar with only one element. the Rowrank should be =0%s","\n");
                    this->_bonds.clear();
                    this->_block = in_tensor;
                    this->_labels.clear();
                    this->_Rowrank = Rowrank;
                }else{
                    std::vector<Bond> bds;
                    for(cytnx_uint64 i=0;i<in_tensor.shape().size();i++){
                        bds.push_back(Bond(in_tensor.shape()[i]));
                    }
                    this->_bonds = bds;
                    this->_block = in_tensor;
                    this->_labels = utils_internal::range_cpu<cytnx_int64>(in_tensor.shape().size());
                    cytnx_error_msg(Rowrank > in_tensor.shape().size(),"[ERROR][Init_by_tensor] Rowrank exceed the rank of Tensor.%s","\n");
                    this->_Rowrank = Rowrank;
                }
            }


    boost::intrusive_ptr<CyTensor_base> DenseCyTensor::permute(const std::vector<cytnx_int64> &mapper,const cytnx_int64 &Rowrank, const bool &by_label){
        boost::intrusive_ptr<CyTensor_base> out = this->clone();
        out->permute_(mapper,Rowrank,by_label);
        return out;
    }
    void DenseCyTensor::permute_(const std::vector<cytnx_int64> &mapper, const cytnx_int64 &Rowrank, const bool& by_label){
        std::vector<cytnx_uint64> mapper_u64;
        if(by_label){
            //cytnx_error_msg(true,"[Developing!]%s","\n");
            std::vector<cytnx_int64>::iterator it;
            for(cytnx_uint64 i=0;i<mapper.size();i++){
                it = std::find(this->_labels.begin(),this->_labels.end(),mapper[i]);
                cytnx_error_msg(it == this->_labels.end(),"[ERROR] label %d does not exist in current CyTensor.\n",mapper[i]);
                mapper_u64.push_back(std::distance(this->_labels.begin(),it));
            }
            
        }else{
            mapper_u64 = std::vector<cytnx_uint64>(mapper.begin(),mapper.end());
        }

        if(this->_is_diag){
            if(Rowrank>=0){
                cytnx_error_msg(Rowrank!=1,"[ERROR] Rowrank should be =1 for CyTensor with is_diag=true%s","\n");
            }
            this->_bonds = vec_map(vec_clone(this->bonds()),mapper_u64);// this will check validity
            this->_labels = vec_map(this->labels(),mapper_u64);
            this->_is_braket_form = this->_update_braket();

        }else{
            
            this->_bonds = vec_map(vec_clone(this->bonds()),mapper_u64);// this will check validity
            this->_labels = vec_map(this->labels(),mapper_u64);
            this->_block.permute_(mapper_u64);
            if(Rowrank>=0){
                cytnx_error_msg((Rowrank>this->_bonds.size()) || (Rowrank < 0),"[ERROR] Rowrank cannot exceed the rank of CyTensor, and should be >=0.%s","\n");
                this->_Rowrank = Rowrank;
            }
            this->_is_braket_form = this->_update_braket();
        }
        
    };


    void DenseCyTensor::print_diagram(const bool &bond_info){
        char *buffer = (char*)malloc(256*sizeof(char));
        
        sprintf(buffer,"-----------------------%s","\n");         std::cout << std::string(buffer);
        sprintf(buffer,"tensor Name : %s\n",this->_name.c_str()); std::cout << std::string(buffer);
        sprintf(buffer,"tensor Rank : %d\n",this->_labels.size());std::cout << std::string(buffer);
        sprintf(buffer,"block_form  : false%s","\n");             std::cout << std::string(buffer);
        sprintf(buffer,"is_diag     : %s\n",this->_is_diag?"True":"False"); std::cout << std::string(buffer);
        sprintf(buffer,"on device   : %s\n",this->device_str().c_str());    std::cout << std::string(buffer);

        cytnx_uint64 Nin = this->_Rowrank;
        cytnx_uint64 Nout = this->_labels.size() - this->_Rowrank;
        cytnx_uint64 vl;
        if(Nin > Nout) vl = Nin;
        else           vl = Nout;

        std::string bks;
        char *l = (char*)malloc(40*sizeof(char));
        char *llbl = (char*)malloc(40*sizeof(char));
        char *r = (char*)malloc(40*sizeof(char));
        char *rlbl = (char*)malloc(40*sizeof(char));
        if(this->is_tag()){
            sprintf(buffer,"braket_form : %s\n",this->_is_braket_form?"True":"False"); std::cout << std::string(buffer);
            sprintf(buffer,"        row               col   %s","\n"); std::cout << std::string(buffer);
            sprintf(buffer,"           ---------------      %s","\n"); std::cout << std::string(buffer);
            for(cytnx_uint64 i=0;i<vl;i++){
                sprintf(buffer,"           |             |     %s","\n"); std::cout << std::string(buffer);
                if(i<Nin){
                    if(this->_bonds[i].type() == bondType::BD_KET) bks = " -->";
                    else                                         bks = "*<--";
                    memset(l,0,sizeof(char)*40);
                    memset(llbl,0,sizeof(char)*40);
                    sprintf(l,"%3d %s",this->_labels[i],bks.c_str());
                    sprintf(llbl,"%-3d",this->_bonds[i].dim()); 
                }else{
                    memset(l,0,sizeof(char)*40);
                    memset(llbl,0,sizeof(char)*40);
                    sprintf(l,"%s","        ");
                    sprintf(llbl,"%s","   "); 
                }
                if(i< Nout){
                    if(this->_bonds[Nin+i].type() == bondType::BD_KET) bks = "<--*";
                    else                                              bks = "--> ";
                    memset(r,0,sizeof(char)*40);
                    memset(rlbl,0,sizeof(char)*40);
                    sprintf(r,"%s %-3d",bks.c_str(),this->_labels[Nin + i]);
                    sprintf(rlbl,"%3d",this->_bonds[Nin + i].dim()); 
                }else{
                    memset(r,0,sizeof(char)*40);
                    memset(rlbl,0,sizeof(char)*40);
                    sprintf(r,"%s","        ");
                    sprintf(rlbl,"%s","   "); 
                }
                sprintf(buffer,"   %s| %s     %s |%s\n",l,llbl,rlbl,r); std::cout << std::string(buffer);

            }
            sprintf(buffer,"           |             |     %s","\n"); std::cout << std::string(buffer);
            sprintf(buffer,"           ---------------     %s","\n"); std::cout << std::string(buffer);

        }else{
            sprintf(buffer,"            -------------      %s","\n"); std::cout << std::string(buffer);
            for(cytnx_uint64 i=0;i<vl;i++){
                if(i == 0) {sprintf(buffer,"           /             \\     %s","\n");std::cout << std::string(buffer);}
                else       {sprintf(buffer,"           |             |     %s","\n"); std::cout << std::string(buffer);}
                
                if(i< Nin){
                    bks = "__";
                    memset(l,0,sizeof(char)*40);
                    memset(llbl,0,sizeof(char)*40);
                    sprintf(l,"%3d %s__",this->_labels[i],bks.c_str());
                    sprintf(llbl,"%-3d",this->_bonds[i].dim());
                }else{
                    memset(l,0,sizeof(char)*40);
                    memset(llbl,0,sizeof(char)*40);
                    sprintf(l,"%s","        ");
                    sprintf(llbl,"%s","   ");
                }
                if(i<Nout){
                    bks = "__";
                    memset(r,0,sizeof(char)*40);
                    memset(rlbl,0,sizeof(char)*40);
                    sprintf(r,"__%s %-3d",bks.c_str(), this->_labels[Nin+i]);
                    sprintf(rlbl,"%3d",this->_bonds[Nin+i].dim());
                }else{
                    memset(r,0,sizeof(char)*40);
                    memset(rlbl,0,sizeof(char)*40);
                    sprintf(r,"%s","        ");
                    sprintf(rlbl,"%s","   ");
                }
                sprintf(buffer,"   %s| %s     %s |%s\n",l,llbl,rlbl,r); std::cout << std::string(buffer);
            } 
            sprintf(buffer,"           \\             /     %s","\n"); std::cout << std::string(buffer);
            sprintf(buffer,"            -------------      %s","\n");  std::cout << std::string(buffer);
        }

        if(bond_info){
            for(cytnx_uint64 i=0; i< this->_bonds.size();i++){
                sprintf(buffer,"lbl:%d ",this->_labels[i]); std::cout << std::string(buffer);
                std::cout << this->_bonds[i] << std::endl;
            }
        }

        //fflush(stdout);
        free(l);
        free(llbl);
        free(r);
        free(rlbl);
        free(buffer);
    }


    void DenseCyTensor::combineBonds(const std::vector<cytnx_int64> &indicators, const bool &permute_back, const bool &by_label){
        cytnx_error_msg(indicators.size() < 2,"[ERROR] the number of bonds to combine must be > 1%s","\n");
        std::vector<cytnx_int64>::iterator it;
        std::vector<cytnx_uint64> idx_mapper;
        if(by_label){
            
            //find the index of label:
            for(cytnx_uint64 i=0;i<indicators.size();i++){
                it = std::find(this->_labels.begin(),this->_labels.end(),indicators[i]);
                cytnx_error_msg(it == this->_labels.end(),"[ERROR] labels not found in current CyTensor%s","\n");
                idx_mapper.push_back(std::distance(this->_labels.begin(),it));
            }

        }else{
            idx_mapper = std::vector<cytnx_uint64>(indicators.begin(),indicators.end());
        }

        ///first permute the Tensor:
        std::vector<cytnx_uint64> old_shape = this->shape();
        
        cytnx_error_msg(this->_is_diag,"[ERROR] cannot combineBond on a is_diag=True CyTensor. suggestion: try CyTensor.to_dense()/to_dense_() first.%s","\n");

        if(permute_back){
            cytnx_uint64 new_Nin = this->_Rowrank;
            //[Fusion tree]>>>
            for(cytnx_uint64 i=1;i<idx_mapper.size();i++){
                if(idx_mapper[i] < this->_Rowrank) new_Nin -=1;
                this->_bonds[idx_mapper[0]].combineBond_(this->_bonds[idx_mapper[i]]);
            }
            //<<<
            ///create mapper for permute
            std::vector<cytnx_uint64> idx_no_combine = utils_internal::range_cpu(this->_labels.size());
            vec_erase_(idx_no_combine,idx_mapper);
            
            std::vector<cytnx_uint64> mapper;
            vec_concatenate_(mapper,idx_mapper,idx_no_combine);
            
            std::vector<cytnx_int64> new_shape; new_shape.push_back(-1);
            for(cytnx_uint64 i=0;i<idx_no_combine.size();i++)
                new_shape.push_back(this->_bonds[idx_no_combine[i]].dim());

            this->_block.permute_(mapper); 

            this->_block.reshape_(new_shape);

            cytnx_int64 f_label = this->_labels[idx_mapper[0]];
            vec_erase_(this->_bonds,std::vector<cytnx_uint64>(idx_mapper.begin()+1,idx_mapper.end()));
            vec_erase_(this->_labels,std::vector<cytnx_uint64>(idx_mapper.begin()+1,idx_mapper.end()));
            //permute back>>
            //find index 
            cytnx_uint64 x = vec_where(this->_labels,f_label);                                
            idx_no_combine = utils_internal::range_cpu(1,this->_labels.size());
            idx_no_combine.insert(idx_no_combine.begin()+x,0);
            this->_block.permute_(idx_no_combine);
            this->_Rowrank = new_Nin;
            
            if(this->is_tag()){
                this->_is_braket_form = this->_update_braket();
            }

        }else{
            //[Fusion tree]>>>
            for(cytnx_uint64 i=1;i<idx_mapper.size();i++){
                this->_bonds[idx_mapper[0]].combineBond_(this->_bonds[idx_mapper[i]]);
            }                  
            //<<<
            std::vector<cytnx_uint64> idx_no_combine = utils_internal::range_cpu(this->_labels.size());
            vec_erase_(idx_no_combine,idx_mapper);
            
            std::vector<cytnx_uint64> mapper;
            std::vector<cytnx_int64> new_shape; 
            if(idx_mapper[0] >= this->_Rowrank){
                std::vector<Bond> new_bonds;
                std::vector<cytnx_int64> new_labels;
                vec_concatenate_(mapper,idx_no_combine,idx_mapper);


                for(cytnx_uint64 i=0;i<idx_no_combine.size();i++){
                    new_shape.push_back(this->_bonds[idx_no_combine[i]].dim());
                    new_bonds.push_back(this->_bonds[idx_no_combine[i]]);
                    new_labels.push_back(this->_labels[idx_no_combine[i]]);
                }
                new_bonds.push_back(this->_bonds[idx_mapper[0]]);
                new_labels.push_back(this->_labels[idx_mapper[0]]);
                new_shape.push_back(-1);                   

                this->_block.permute_(mapper);
                this->_block.reshape_(new_shape); 
                
                this->_bonds = new_bonds;
                this->_labels = new_labels;
                this->_Rowrank = this->_labels.size()-1;


            }else{
                std::vector<Bond> new_bonds;
                std::vector<cytnx_int64> new_labels;
                vec_concatenate_(mapper,idx_mapper,idx_no_combine);
                
                new_bonds.push_back(this->_bonds[idx_mapper[0]]);
                new_labels.push_back(this->_labels[idx_mapper[0]]);
                new_shape.push_back(-1);                   
                for(cytnx_uint64 i=0;i<idx_no_combine.size();i++){
                    new_shape.push_back(this->_bonds[idx_no_combine[i]].dim());
                    new_bonds.push_back(this->_bonds[idx_no_combine[i]]);
                    new_labels.push_back(this->_labels[idx_no_combine[i]]);
                }

                this->_block.permute_(mapper);
                this->_block.reshape_(new_shape);
                
                this->_bonds = new_bonds;
                this->_labels = new_labels;
                this->_Rowrank = 1;


            }

            if(this->is_tag()){
                this->_is_braket_form = this->_update_braket();
            }
        }//permute_back

    }

    boost::intrusive_ptr<CyTensor_base> DenseCyTensor::to_dense(){
        cytnx_error_msg(!(this->_is_diag),"[ERROR] to_dense can only operate on CyTensor with is_diag = True.%s","\n");
        DenseCyTensor *tmp = this->clone_meta();
        tmp->_block = cytnx::linalg::Diag(this->_block);
        tmp->_is_diag = false;
        boost::intrusive_ptr<CyTensor_base> out(tmp);
        return out;
    }
    void DenseCyTensor::to_dense_(){
                cytnx_error_msg(!(this->_is_diag),"[ERROR] to_dense_ can only operate on CyTensor with is_diag = True.%s","\n");
                this->_block = cytnx::linalg::Diag(this->_block);
                this->_is_diag = false;
    }

    boost::intrusive_ptr<CyTensor_base> DenseCyTensor::contract(const boost::intrusive_ptr<CyTensor_base> &rhs){
        //checking :
        cytnx_error_msg(rhs->is_blockform() ,"[ERROR] cannot contract non-symmetry CyTensor with symmetry CyTensor%s","\n");
        cytnx_error_msg(this->is_tag() != rhs->is_tag(), "[ERROR] cannot contract tagged CyTensor with untagged CyTensor.%s","\n");
        //cytnx_error_msg(this->is_diag() != rhs->is_diag(),"[ERROR] cannot contract a diagonal tensor with non-diagonal tensor. [suggestion:] call CyTensor.to_dense/to_dense_ first%s","\n");
        //get common labels:    
        std::vector<cytnx_int64> comm_labels;
        std::vector<cytnx_uint64> comm_idx1,comm_idx2;
        vec_intersect_(comm_labels,this->labels(),rhs->labels(),comm_idx1,comm_idx2);
        //output instance:
        DenseCyTensor *tmp = new DenseCyTensor();
            
        tmp->_bonds.clear();
        tmp->_labels.clear();


        if(comm_idx1.size() == 0){
            //process meta
            vec_concatenate_(tmp->_labels,this->labels(),rhs->labels());
            
            // these two cannot omp parallel, due to intrusive_ptr
            for(cytnx_uint64 i=0; i<this->_bonds.size();i++)
                tmp->_bonds.push_back(this->_bonds[i].clone());
            for(cytnx_uint64 i=0; i<rhs->_bonds.size();i++)
                tmp->_bonds.push_back(rhs->_bonds[i].clone());

            tmp->_is_tag = this->is_tag();
            tmp->_Rowrank = this->Rowrank() + rhs->Rowrank();

            if((this->is_diag() == rhs->is_diag()) && this->is_diag()){
                tmp->_block = linalg::Kron(this->_block, rhs->get_block_());
                tmp->_block.reshape_({-1});
                tmp->_is_diag = true;
            }else{
                Tensor tmpL,tmpR;
                if(this->is_diag()) tmpL = linalg::Diag(this->_block);
                else{ 
                    if(this->_block.is_contiguous())
                        tmpL = this->_block;
                    else
                        tmpL = this->_block.contiguous();
                }

                if(rhs->is_diag()) tmpR = linalg::Diag(rhs->get_block_());
                else{ 
                    if(rhs->get_block_().is_contiguous())
                        tmpR =  rhs->get_block_(); // share view!!
                    else
                        tmpR =  rhs->get_block_().contiguous();
                }
                std::vector<cytnx_int64> old_shape_L(tmpL.shape().begin(),tmpL.shape().end());
                //vector<cytnx_int64> old_shape_R(tmpR.shape().begin(),tmpR.shape().end());
                std::vector<cytnx_int64> shape_L = vec_concatenate(old_shape_L,std::vector<cytnx_int64>(tmpR.shape().size(),1)); 
                //vector<cytnx_int64> shape_R = vec_concatenate(vector<cytnx_int64>(old_shape_L.size(),1),old_shape_R);
                tmpL.reshape_(shape_L);
                //tmpR.reshape_(shape_R);
                tmp->_block = linalg::Kron(tmpL,tmpR,false,true);
                tmpL.reshape_(old_shape_L);
                //tmpR.reshape_(old_shapeR);
                tmp->_is_diag = false;

            }
            tmp->_is_braket_form = tmp->_update_braket();

        }else{

            // if tag, checking bra-ket matching!
            if(this->is_tag()){
                for(int i=0;i<comm_idx1.size();i++)
                    cytnx_error_msg(this->_bonds[comm_idx1[i]].type() == rhs->_bonds[comm_idx2[i]].type(),"[ERROR][DenseCyTensor][contract] cannot contract common label: <%d> @ self bond#%d & rhs bond#%d, BRA-KET mismatch!%s",this->labels()[comm_idx1[i]],comm_idx1[i],comm_idx2[i],"\n");
            }

            //process meta
            std::vector<cytnx_uint64> non_comm_idx1 = vec_erase(utils_internal::range_cpu(this->rank()),comm_idx1);
            std::vector<cytnx_uint64> non_comm_idx2 = vec_erase(utils_internal::range_cpu(rhs->rank()),comm_idx2);
                
            vec_concatenate_(tmp->_labels,vec_clone(this->_labels,non_comm_idx1),vec_clone(rhs->_labels,non_comm_idx2));
            
            // these two cannot omp parallel, due to intrusive_ptr
            for(cytnx_uint64 i=0; i<non_comm_idx1.size();i++)
                tmp->_bonds.push_back(this->_bonds[non_comm_idx1[i]].clone());
            for(cytnx_uint64 i=0; i<non_comm_idx2.size();i++)
                tmp->_bonds.push_back(rhs->_bonds[non_comm_idx2[i]].clone());
            
            tmp->_is_tag = this->is_tag();
            tmp->_Rowrank = this->Rowrank() + rhs->Rowrank();
            for(cytnx_uint64 i=0; i<comm_idx1.size();i++)
                if(comm_idx1[i] < this->_Rowrank) tmp->_Rowrank--;
            for(cytnx_uint64 i=0;i<comm_idx2.size();i++)
                if(comm_idx2[i] < rhs->_Rowrank) tmp->_Rowrank--;

            if((this->is_diag() == rhs->is_diag()) && this->is_diag()){
                //diag x diag:
                if(tmp->_Rowrank!=0){
                    tmp->_block = this->_block * rhs->get_block_();
                }else{
                    tmp->_block = linalg::Vectordot(this->_block,rhs->get_block_());
                }
                tmp->_is_diag = true;
            }else{
                // diag x dense:
                Tensor tmpL,tmpR;
                if(this->is_diag()) tmpL = linalg::Diag(this->_block);
                else tmpL = this->_block; 
                if(rhs->is_diag()) tmpR = linalg::Diag(rhs->get_block_());
                else tmpR =  rhs->get_block_(); // share view!!
                tmp->_block = linalg::Tensordot(tmpL,tmpR,comm_idx1,comm_idx2);
                tmp->_is_diag = false;
            }
            tmp->_is_braket_form = tmp->_update_braket();
 
        }// check if no common index
                    
        boost::intrusive_ptr<CyTensor_base> out(tmp);
        return out;

    }
    
    void DenseCyTensor::Trace_(const cytnx_int64 &a, const cytnx_int64 &b, const bool &by_label){

        // 1) from label to indx. 
        cytnx_uint64 ida, idb;

        if(by_label){
            ida = vec_where(this->_labels,a);
            idb = vec_where(this->_labels,b);
        }else{
            cytnx_error_msg(a < 0 || b < 0,"[ERROR] invalid index a, b%s","\n");
            cytnx_error_msg(a >= this->rank() || b>= this->rank(),"[ERROR] index out of bound%s","\n");
            ida=a;idb=b;
        }


        // check if indices are the same:
        cytnx_error_msg(ida == idb, "[ERROR][DenseCyTensor::Trace_] index a and index b should not be the same.%s","\n");

        // check dimension:
        cytnx_error_msg(this->_bonds[ida].dim()!= this->_bonds[idb].dim(),"[ERROR][DenseCyTensor::Trace_] The dimension of two bond for trace does not match!%s","\n");
        
        // check bra-ket if tagged 
        if(this->is_braket_form()){

            //check if it is the same species:
            if(this->_bonds[ida].type() == this->_bonds[idb].type()){
                cytnx_error_msg(true,"[ERROR][DenseCyTensor::Trace_] BD_BRA can only contract with BD_KET.%s","\n");
            }

        }


        // trace the block:
        if(this->_is_diag){
            cytnx_error_msg(true,"[Error] We need linalg.Sum!%s","\n");
        }else{
            this->_block = this->_block.Trace(ida,idb);
        }
        

        // update Rowrank:
        cytnx_int64 tmpRk = this->_Rowrank;
        if(ida < tmpRk) this->_Rowrank--;
        if(idb < tmpRk) this->_Rowrank--;

        // remove the bound, labels:
        if(ida > idb) std::swap(ida,idb);
        this->_bonds.erase(this->_bonds.begin()+idb);
        this->_bonds.erase(this->_bonds.begin()+ida);
        this->_labels.erase(this->_labels.begin()+idb);
        this->_labels.erase(this->_labels.begin()+ida);

    }

    void DenseCyTensor::Transpose_(){
        std::vector<cytnx_int64> new_permute  = vec_concatenate(vec_range<cytnx_int64>(this->Rowrank(),this->rank()),vec_range<cytnx_int64>(0,this->Rowrank()));
        this->permute_(new_permute);
        if(this->is_tag()){
            this->_Rowrank = this->rank() - this->_Rowrank;
            for(int i=0;i<this->rank();i++){
                this->_bonds[i].set_type((this->_bonds[i].type()==BD_KET)?BD_BRA:BD_KET);
            }
            this->_is_braket_form = this->_update_braket();
        }else{
            this->_Rowrank = this->rank() - this->_Rowrank;
        }
    }


    void DenseCyTensor::_save_dispatch(std::fstream &f) const{
        this->_block._Save(f); 
    }
    void DenseCyTensor::_load_dispatch(std::fstream &f){
        this->_block._Load(f); 
    }


}
