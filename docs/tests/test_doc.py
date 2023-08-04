import sys
#CYTNX_ROOT = '/home/hunghaoti/Libraries/Cytnx_lib'
sys.path.insert(0, CYTNX_ROOT)
import cytnx
import pytest
import time

ex_code_path = '../code/python/'
test_codes_path = ex_code_path + 'doc_codes/'
output_path = ex_code_path + 'outputs/'

def needGPUTest(obj):
    #if need to test cuda, set us True
    obj.__test__ = False
    return obj

def get_redirect_code(test_name):
    str_code = '''
captured = capsys.readouterr()
if captured.out:
    text_file = open(output_path + \'''' + test_name + '''.out', 'w')
    text_file.write(captured.out)
    text_file.close()
'''
    return str_code

def excute_all(test_names, capsys):
    codes = ''
    for test_name in test_names:
        code = open(test_codes_path + test_name + '.py').read() + \
               get_redirect_code(test_name)
        codes = codes + code
    exec(codes)


def excute_and_output(test_name, capsys):
    exec(open(test_codes_path + test_name + '.py').read())
    captured = capsys.readouterr()
    text_file = open(output_path + test_name + '.out', 'w')
    text_file.write(captured.out)
    text_file.close()

# test case main

def test_guide_behavior_assign(capsys):
    excute_and_output('guide_behavior_assign', capsys)

def test_guide_behavior_clone(capsys):
    excute_and_output('guide_behavior_clone', capsys)

def test_guide_Device_Ncpus(capsys):
    excute_and_output('guide_Device_Ncpus', capsys)

def test_guide_Device_Ngpus(capsys):
    excute_and_output('guide_Device_Ngpus', capsys)

def test_guide_Device_property(capsys):
    excute_and_output('guide_Device_property', capsys)

def test_guide_basic_obj_Tensor_1_create_zeros(capsys):
    excute_and_output('guide_basic_obj_Tensor_1_create_zeros', capsys)

def test_guide_basic_obj_Tensor_1_create_diff_ways(capsys):
    excute_and_output('guide_basic_obj_Tensor_1_create_diff_ways', capsys)

def test_guide_basic_obj_Tensor_1_create_rand(capsys):
    excute_and_output('guide_basic_obj_Tensor_1_create_rand', capsys)

@needGPUTest
def test_guide_basic_obj_Tensor_1_create_zeros_cuda(capsys):
    excute_and_output('guide_basic_obj_Tensor_1_create_zeros_cuda', capsys)

def test_guide_basic_obj_Tensor_1_create_astype(capsys):
    excute_and_output('guide_basic_obj_Tensor_1_create_astype', capsys)

@needGPUTest
def test_guide_basic_obj_Tensor_1_create_to(capsys):
    excute_and_output('guide_basic_obj_Tensor_1_create_to', capsys)

def test_guide_basic_obj_Tensor_1_create_from_storage(capsys):
    excute_and_output('guide_basic_obj_Tensor_1_create_from_storage', capsys)

def test_guide_basic_obj_Tensor_2_manip_reshape(capsys):
    excute_and_output('guide_basic_obj_Tensor_2_manip_reshape', capsys)

def test_guide_basic_obj_Tensor_2_manip_reshape_(capsys):
    excute_and_output('guide_basic_obj_Tensor_2_manip_reshape_', capsys)

def test_guide_basic_obj_Tensor_2_manip_permute(capsys):
    excute_and_output('guide_basic_obj_Tensor_2_manip_permute', capsys)

def test_guide_basic_obj_Tensor_2_manip_contiguous(capsys):
    excute_and_output('guide_basic_obj_Tensor_2_manip_contiguous', capsys)

def test_guide_basic_obj_Tensor_3_access_slice_get(capsys):
    excute_and_output('guide_basic_obj_Tensor_3_access_slice_get', capsys)

def test_guide_basic_obj_Tensor_3_access_item(capsys):
    excute_and_output('guide_basic_obj_Tensor_3_access_item', capsys)

def test_guide_basic_obj_Tensor_3_access_slice_set(capsys):
    excute_and_output('guide_basic_obj_Tensor_3_access_slice_set', capsys)

def test_guide_basic_obj_Tensor_4_arithmetic_tensor_scalar(capsys):
    excute_and_output('guide_basic_obj_Tensor_4_arithmetic_tensor_scalar', capsys)

def test_guide_basic_obj_Tensor_4_arithmetic_tensor_tensor(capsys):
    excute_and_output('guide_basic_obj_Tensor_4_arithmetic_tensor_tensor', capsys)

def test_guide_basic_obj_Tensor_5_numpy_cytnx2numpy(capsys):
    excute_and_output('guide_basic_obj_Tensor_5_numpy_cytnx2numpy', capsys)

def test_guide_basic_obj_Tensor_5_numpy_numpy2cytnx(capsys):
    excute_and_output('guide_basic_obj_Tensor_5_numpy_numpy2cytnx', capsys)

def test_guide_basic_obj_Tensor_6_app_scalar(capsys):
    excute_and_output('guide_basic_obj_Tensor_6_app_scalar', capsys)

def test_guide_basic_obj_Tensor_6_app_tensor(capsys):
    excute_and_output('guide_basic_obj_Tensor_6_app_tensor', capsys)

def test_guide_basic_obj_Tensor_7_io_Save(capsys):
    excute_and_output('guide_basic_obj_Tensor_7_io_Save', capsys)

def test_guide_basic_obj_Tensor_7_io_Load(capsys):
    excute_and_output('guide_basic_obj_Tensor_7_io_Load', capsys)

def test_guide_basic_obj_Tensor_8_cp_assign(capsys):
    excute_and_output('guide_basic_obj_Tensor_8_cp_assign', capsys)

def test_guide_basic_obj_Tensor_8_cp_clone(capsys):
    excute_and_output('guide_basic_obj_Tensor_8_cp_clone', capsys)

def test_guide_basic_obj_Tensor_8_cp_permute(capsys):
    test_names = ['guide_basic_obj_Tensor_8_cp_permute-1', \
                  'guide_basic_obj_Tensor_8_cp_permute-2', \
                  'guide_basic_obj_Tensor_8_cp_permute-3']
    excute_all(test_names, capsys)

def test_guide_basic_obj_Tensor_8_cp_contiguous(capsys):
    test_names = ['guide_basic_obj_Tensor_8_cp_contiguous-1', \
                  'guide_basic_obj_Tensor_8_cp_contiguous-2']
    excute_all(test_names, capsys)

def test_guide_basic_obj_Storage_1_create_create(capsys):
    excute_and_output('guide_basic_obj_Storage_1_create_create', capsys)

def test_guide_basic_obj_Storage_1_create_astype(capsys):
    excute_and_output('guide_basic_obj_Storage_1_create_astype', capsys)

@needGPUTest
def test_guide_basic_obj_Storage_1_create_to(capsys):
    excute_and_output('guide_basic_obj_Storage_1_create_to', capsys)

def test_guide_basic_obj_Storage_1_create_get_storage(capsys):
    excute_and_output('guide_basic_obj_Storage_1_create_get_storage', capsys)

def test_guide_basic_obj_Storage_1_create_contiguous_check(capsys):
    excute_and_output('guide_basic_obj_Storage_1_create_contiguous_check', capsys)

def test_user_guide_4_2_1_ex1(capsys):
    excute_and_output('guide_basic_obj_Storage_2_access_access', capsys)

def test_guide_basic_obj_Storage_3_expand_append(capsys):
    excute_and_output('guide_basic_obj_Storage_3_expand_append', capsys)

def test_guide_basic_obj_Storage_3_expand_resize(capsys):
    excute_and_output('guide_basic_obj_Storage_3_expand_resize', capsys)

def test_guide_basic_obj_Storage_5_io_Save(capsys):
    excute_and_output('guide_basic_obj_Storage_5_io_Save', capsys)

def test_guide_basic_obj_Storage_5_io_Load(capsys):
    excute_and_output('guide_basic_obj_Storage_5_io_Load', capsys)

def test_guide_basic_obj_Storage_5_io_from_to_file(capsys):
    excute_and_output('guide_basic_obj_Storage_5_io_from_to_file', capsys)

def test_guide_uniten_print(capsys):
    test_names = ['guide_uniten_print_init', \
                  'guide_uniten_print_print_diagram', \
                  'guide_uniten_print_set_name', \
                  'guide_uniten_print_print_block', \
                  'guide_uniten_print_sym_print', \
                  'guide_uniten_print_sym_print_block']
    excute_all(test_names, capsys)

def test_guide_uniten_create_from_tensor(capsys):
    excute_and_output('guide_uniten_create_from_tensor', capsys)

def test_guide_uniten_create_complex(capsys):
    test_names = ['guide_uniten_create_complex', \
                  'guide_uniten_create_print_diagram']
    excute_all(test_names, capsys)

def test_guide_uniten_create_scratch(capsys):
    excute_and_output('guide_uniten_create_scratch', capsys)

def test_guide_uniten_labels_relabel(capsys):
    test_names = ['guide_uniten_labels_relabel_', \
                  'guide_uniten_labels_relabel']
    excute_all(test_names, capsys)

def test_guide_uniten_bond_create(capsys):
    excute_and_output('guide_uniten_bond_create', capsys)

def test_guide_uniten_bond_redirect(capsys):
    excute_and_output('guide_uniten_bond_redirect', capsys)

def test_guide_uniten_bond_symobj(capsys):
    excute_and_output('guide_uniten_bond_symobj', capsys)

def test_user_guide_7_4_2_3(capsys):
    test_names = ['guide_uniten_bond_sym_bond', \
                 'guide_uniten_bond_multi_sym_bond', \
                 'guide_uniten_bond_combine', \
                 'guide_uniten_bond_combine_no_grp']
    excute_all(test_names, capsys)

def test_guide_uniten_tagged(capsys):
    with pytest.raises(Exception) as e_info:
        test_names = ['guide_uniten_tagged_init', \
                      'guide_uniten_tagged_contract']
        excute_all(test_names, capsys)
    print(str(e_info.value))
    test_name = 'guide_uniten_tagged_contract'
    exec(get_redirect_code(test_name))

def test_guide_uniten_symmetric(capsys):
    test_names = ['guide_uniten_symmetric_create', \
                  'guide_uniten_symmetric_print_blocks']
    excute_all(test_names, capsys)

def test_guide_uniten_blocks_get_block(capsys):
    excute_and_output('guide_uniten_blocks_get_block', capsys)

def test_guide_uniten_blocks_put_get_block(capsys):
    test_names = ['guide_uniten_blocks_init', \
                  'guide_uniten_blocks_get_block_qidx', \
                  'guide_uniten_blocks_get_block_bkidx', \
                  'guide_uniten_blocks_get_blocks_', \
                  'guide_uniten_blocks_put_block_qidx', \
                  'guide_uniten_blocks_put_block_bkidx']
    excute_all(test_names, capsys)

def test_guide_uniten_elements_at_get(capsys):
    excute_and_output('guide_uniten_elements_at_get', capsys)

def test_guide_uniten_elements_at_set(capsys):
    excute_and_output('guide_uniten_elements_at_set', capsys)

def test_guide_uniten_elements_sym(capsys):
    with pytest.raises(Exception) as e_info:
        test_names = ['guide_uniten_elements_init_sym', \
                      'guide_uniten_elements_at_qidx', \
                      'guide_uniten_elements_at_non_exist']
        excute_all(test_names, capsys)
    print(str(e_info.value))
    test_name = 'guide_uniten_elements_at_non_exist'
    exec(get_redirect_code(test_name))

def test_guide_uniten_elements_exists(capsys):
    test_names = ['guide_uniten_elements_init_sym', \
                  'guide_uniten_elements_exists']
    excute_all(test_names, capsys)

def test_guide_uniten_manipulation_permute(capsys):
    excute_and_output('guide_uniten_manipulation_permute', capsys)

def test_guide_uniten_manipulation_reshape(capsys):
    excute_and_output('guide_uniten_manipulation_reshape', capsys)

def test_guide_uniten_manipulation_combine(capsys):
    excute_and_output('guide_uniten_manipulation_combine', capsys)

def test_guide_xlinalg_Svd(capsys):
    test_names = ['guide_xlinalg_Svd', \
                  'guide_xlinalg_Svd_verify']
    excute_all(test_names, capsys)

def test_guide_uniten_io_Save(capsys):
    excute_and_output('guide_uniten_io_Save', capsys)
    time.sleep(0.01) # wait to save field, then next test can load

def test_guide_uniten_io_Load(capsys):
    excute_and_output('guide_uniten_io_Load', capsys)

def test_guide_contraction_network_launch(capsys):
    test_names = ['guide_contraction_network_PutUniTensor', \
                  'guide_contraction_network_launch']
    excute_all(test_names, capsys)

def test_guide_contraction_network_FromString(capsys):
    excute_and_output('guide_contraction_network_FromString', capsys)

def test_guide_contraction_network_label_ord(capsys):
    test_names = ['guide_contraction_network_label_ord-1', \
                  'guide_contraction_network_label_ord-2', \
                  'guide_contraction_network_label_ord-3']
    excute_all(test_names, capsys)

def test_guide_contraction_contract_Contract(capsys):
    excute_and_output('guide_contraction_contract_Contract', capsys)

def test_guide_contraction_contract_relabes(capsys):
    excute_and_output('guide_contraction_contract_relabels', capsys)

def test_guide_contraction_contract_Contracts(capsys):
    excute_and_output('guide_contraction_contract_Contracts', capsys)

def test_guide_contraction_ncon_ncon(capsys):
    excute_and_output('guide_contraction_ncon_ncon', capsys)

def test_guide_itersol_LinOp_Dot(capsys):
    excute_and_output('guide_itersol_LinOp_Dot', capsys)

def test_guide_itersol_LinOp_inherit(capsys):
    excute_and_output('guide_itersol_LinOp_inherit', capsys)

def test_guide_itersol_LinOp_matvec(capsys):
    test_names = ['guide_itersol_LinOp_matvec', \
                  'guide_itersol_LinOp_demo']
    excute_all(test_names, capsys)

def test_guide_itersol_LinOp_sparse_mv(capsys):
    excute_and_output('guide_itersol_LinOp_sparse_mv', capsys)

def test_guide_itersol_LinOp_sparse_mv_elem(capsys):
    excute_and_output('guide_itersol_LinOp_sparse_mv_elem', capsys)

def test_guide_itersol_Lanczos_Lanczos(capsys):
    excute_and_output('guide_itersol_Lanczos_Lanczos', capsys)
