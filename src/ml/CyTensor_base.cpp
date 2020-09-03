#include "ml/CyTensor.hpp"
#include "ml/xlinalg.hpp"
#include "torcyx.hpp"
#include <string>
#include "utils/utils.hpp"

namespace torcyx{
        using cytnx::vec_unique;
        //====================================================    
        std::string CyTensorType_class::getname(const int &ut_type){
            if(ut_type==this->Void){
                return std::string("Void (un-initialize CyTensor");
            }else if(ut_type==this->Dense){
                return std::string("Dense");
            }else if(ut_type==this->Sparse){
                return std::string("Sparse (block-form)");
            }else{
                cytnx_error_msg(true,"%s\n","[ERROR] invalid ut_type");
                return std::string("");
            }
            // extend more in here!!
        }
        CyTensorType_class CTenType;

        //===================================================
            
            //void CyTensor_base::Init(const std::vector<Bond> &bonds, const std::vector<cytnx_int64> &in_labels, const cytnx_int64 &rowrank,const unsigned int &dtype,const int &device,const bool &is_diag){
            //    cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            //}
            void CyTensor_base::Init(const std::vector<Bond> &bonds, const std::vector<cytnx_int64> &in_labels, const cytnx_int64 &rowrank, const bool &is_diag, const torch::TensorOptions &options){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            void CyTensor_base::Init_by_Tensor(const torch::Tensor &in_tensor, const cytnx_uint64 &rowrank, const bool &is_diag){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            std::vector<cytnx_int64> CyTensor_base::shape() const{
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
                return std::vector<cytnx_int64>();
            }
            bool      CyTensor_base::is_blockform() const{
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
                return false;
            }
            bool     CyTensor_base::is_contiguous() const{
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
                return false;
            }
            torch::TensorOptions  CyTensor_base::options() const{
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
                return torch::TensorOptions();
            }

            
            torch::Tensor CyTensor_base::get_block(const cytnx_uint64 &idx) const{
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
                return torch::Tensor();
            }
            torch::Tensor CyTensor_base::get_block(const std::vector<cytnx_int64> &qnum) const{
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
                return torch::Tensor();
            }
            // return a share view of block, this only work for non-symm tensor.
            const torch::Tensor& CyTensor_base::get_block_(const cytnx_uint64 &idx) const{
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
                torch::Tensor t;
                return t;
            }
            // return a share view of block, this only work for non-symm tensor.
            torch::Tensor& CyTensor_base::get_block_(const cytnx_uint64 &idx){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
                torch::Tensor t;
                return t;
            }
            const torch::Tensor& CyTensor_base::get_block_(const std::vector<cytnx_int64> &qnum) const{
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
                torch::Tensor t;
                return t;
            }
            torch::Tensor& CyTensor_base::get_block_(const std::vector<cytnx_int64> &qnum){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
                torch::Tensor t;
                return t;
            }
            std::vector<torch::Tensor> CyTensor_base::get_blocks() const{
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
                return std::vector<torch::Tensor>();
            }
            const std::vector<torch::Tensor>& CyTensor_base::get_blocks_() const{
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
                return std::vector<torch::Tensor>();
            }
            std::vector<torch::Tensor>& CyTensor_base::get_blocks_(){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
                std::vector<torch::Tensor> t;
                return t;
            }


            void CyTensor_base::print_diagram(const bool &bond_info){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }

            void CyTensor_base::to_(const torch::Device &device, const bool &non_blocking){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            boost::intrusive_ptr<CyTensor_base> CyTensor_base::to(const torch::Device &device, const bool &non_blocking){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
                return nullptr;
            }
            boost::intrusive_ptr<CyTensor_base> CyTensor_base::clone() const{
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
                return nullptr;
            }
            torch::Device  CyTensor_base::device() const{
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
                return torch::kCPU;
            }
            /*
            void CyTensor_base::set_labels(const std::vector<cytnx_int64> &new_labels){
                    cytnx_error_msg(new_labels.size()!=this->_labels.size(),"[ERROR][set_labels][CyTensor] %s\n","the len(new_labels) does not match the rank of the UniTenosr");
                    //check duplicate:
                    std::vector<cytnx_int64> tmp = vec_unique<cytnx_int64>(new_labels);
                    cytnx_error_msg(tmp.size() != new_labels.size(), "[ERROR][set_labels][CyTensor] %s\n","the input labels cannot contain duplicated element(s).");
                    this->_labels = new_labels;
            }
             
            void CyTensor_base::set_rowrank(const cytnx_uint64 &new_rowrank){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            unsigned int  CyTensor_base::dtype() const{
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
                return 0;
            }
            std::string      CyTensor_base::dtype_str() const{
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
                return std::string("");
            }
            std::string     CyTensor_base::device_str() const{
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
                return std::string("");
            }
            boost::intrusive_ptr<CyTensor_base> CyTensor_base::permute(const std::vector<cytnx_int64> &mapper,const cytnx_int64 &rowrank, const bool &by_label){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
                return nullptr;
            }
            void CyTensor_base::permute_(const std::vector<cytnx_int64> &mapper, const cytnx_int64 &rowrank, const bool &by_label){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }

            boost::intrusive_ptr<CyTensor_base> CyTensor_base::contiguous_(){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            boost::intrusive_ptr<CyTensor_base> CyTensor_base::contiguous(){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
                boost::intrusive_ptr<CyTensor_base> out(new CyTensor_base());
                return out;
            }            

            void CyTensor_base::put_block(const Tensor &in, const cytnx_uint64 &idx){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            void CyTensor_base::put_block_(Tensor &in, const cytnx_uint64 &idx){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            void CyTensor_base::put_block(const Tensor &in, const std::vector<cytnx_int64> &qnum){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            void CyTensor_base::put_block_(Tensor &in, const std::vector<cytnx_int64> &qnum){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            void CyTensor_base::set(const std::vector<Accessor> &accesssors, const Tensor &rhs){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            boost::intrusive_ptr<CyTensor_base> CyTensor_base::get(const std::vector<Accessor> &accesssors){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
                return nullptr;
            }
            
           
            void CyTensor_base::reshape_(const std::vector<cytnx_int64> &new_shape, const cytnx_uint64 &rowrank){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");

            }
            boost::intrusive_ptr<CyTensor_base> CyTensor_base::reshape(const std::vector<cytnx_int64> &new_shape, const cytnx_uint64 &rowrank){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
                return nullptr;
            }
            
            boost::intrusive_ptr<CyTensor_base> CyTensor_base::to_dense(){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
                return nullptr;
            }
            void CyTensor_base::to_dense_(){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            void CyTensor_base::combineBonds(const std::vector<cytnx_int64> &indicators, const bool &permute_back, const bool &by_labels){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            boost::intrusive_ptr<CyTensor_base> CyTensor_base::contract(const boost::intrusive_ptr<CyTensor_base>&rhs){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
                return nullptr;
            }
            std::vector<Bond> CyTensor_base::getTotalQnums(const bool &physical){

                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
                return std::vector<Bond>();
            }

            boost::intrusive_ptr<CyTensor_base> CyTensor_base::Conj(){

                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
                return nullptr;
            }
            void CyTensor_base::Conj_(){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }

            boost::intrusive_ptr<CyTensor_base> CyTensor_base::Trace(const cytnx_int64 &a, const cytnx_int64 &b, const bool &by_label){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
                return nullptr;
            }
            void CyTensor_base::Trace_(const cytnx_int64 &a, const cytnx_int64 &b, const bool &by_label){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }

            void CyTensor_base::Transpose_(){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            boost::intrusive_ptr<CyTensor_base> CyTensor_base::Transpose(){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
                return nullptr;
            }

            void CyTensor_base::Dagger_(){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            boost::intrusive_ptr<CyTensor_base> CyTensor_base::Dagger(){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
                return nullptr;
            }


            void CyTensor_base::tag(){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }

            cytnx_complex128& CyTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_complex128 &aux){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            cytnx_complex64& CyTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_complex64 &aux){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            cytnx_double& CyTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_double &aux){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            cytnx_float& CyTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_float &aux){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            cytnx_uint64& CyTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_uint64 &aux){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            cytnx_int64& CyTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_int64 &aux){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            cytnx_uint32& CyTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_uint32 &aux){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            cytnx_int32& CyTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_int32 &aux){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            cytnx_uint16& CyTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_uint16 &aux){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            cytnx_int16& CyTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_int16 &aux){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            const cytnx_complex128& CyTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_complex128 &aux) const{
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            const cytnx_complex64& CyTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_complex64 &aux) const{
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            const cytnx_double& CyTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_double &aux) const{
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            const cytnx_float& CyTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_float &aux) const{
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            const cytnx_uint64& CyTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_uint64 &aux) const{
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            const cytnx_int64& CyTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_int64 &aux) const {
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            const cytnx_uint32& CyTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_uint32 &aux) const {
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            const cytnx_int32& CyTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_int32 &aux) const {
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            const cytnx_uint16& CyTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_uint16 &aux) const{
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            const cytnx_int16& CyTensor_base::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_int16 &aux) const {
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }

            
            bool CyTensor_base::elem_exists(const std::vector<cytnx_uint64> &locator) const{
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }

            void CyTensor_base::_save_dispatch(std::fstream &f) const{
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }

            void CyTensor_base::_load_dispatch(std::fstream &f){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }

            void CyTensor_base::truncate_(const cytnx_int64 &bond_idx, const cytnx_uint64 &dim, const bool &by_label){
                cytnx_error_msg(true,"[ERROR] fatal internal, cannot call on a un-initialize CyTensor_base%s","\n");
            }
            */
            //-----------------------------------------
            
            std::ostream& operator<<(std::ostream& os, const CyTensor &in){
                char* buffer = (char*)malloc(sizeof(char)*256);
                sprintf(buffer,"Tensor name: %s\n",in.name().c_str()); os << std::string(buffer);
                if(!in.is_tag()) 
                    sprintf(buffer,"braket_form : %s\n", in.is_braket_form()?"True":"False"); os << std::string(buffer);
                
                if(in.is_blockform()){
                    std::vector<torch::Tensor> tmp = in.get_blocks();
                    for(cytnx_uint64 i=0;i<tmp.size();i++)
                        os << tmp[i] << std::endl;
                    
                }else{
                    sprintf(buffer,"is_diag    : %s\n",in.is_diag()?"True":"False"); os << std::string(buffer);
                    torch::Tensor tmp = in.get_block();
                    os << tmp << std::endl;     
                }         
                free(buffer);
                return os;
            
            }
            /*
            CyTensor Contract(const CyTensor &inL, const CyTensor &inR){
                return inL.contract(inR);
            }
            */
            
}//cytnx

