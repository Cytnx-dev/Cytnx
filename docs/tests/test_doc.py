import sys
CYTNX_ROOT = '/home/hunghaoti/Libraries/Cytnx_lib'
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

def test_user_guide_1_1_ex1(capsys):
    excute_and_output('user_guide_1_1_ex1', capsys)

def test_user_guide_1_2_ex1(capsys):
    excute_and_output('user_guide_1_2_ex1', capsys)

def test_user_guide_2_1_ex1(capsys):
    excute_and_output('user_guide_2_1_ex1', capsys)

def test_user_guide_2_2_ex1(capsys):
    excute_and_output('user_guide_2_2_ex1', capsys)

def test_user_guide_2_3_ex1(capsys):
    excute_and_output('user_guide_2_3_ex1', capsys)

def test_user_guide_3_1_1_ex1(capsys):
    excute_and_output('user_guide_3_1_1_ex1', capsys)

def test_user_guide_3_1_2_ex1(capsys):
    excute_and_output('user_guide_3_1_2_ex1', capsys)

@needGPUTest
def test_user_guide_3_1_3_ex1(capsys):
    excute_and_output('user_guide_3_1_3_ex1', capsys)

def test_user_guide_3_1_4_ex1(capsys):
    excute_and_output('user_guide_3_1_4_ex1', capsys)

@needGPUTest
def test_user_guide_3_1_5_ex1(capsys):
    excute_and_output('user_guide_3_1_5_ex1', capsys)

def test_user_guide_3_1_6_ex1(capsys):
    excute_and_output('user_guide_3_1_6_ex1', capsys)

def test_user_guide_3_2_1_ex1(capsys):
    excute_and_output('user_guide_3_2_1_ex1', capsys)

def test_user_guide_3_2_1_ex2(capsys):
    excute_and_output('user_guide_3_2_1_ex2', capsys)

def test_user_guide_3_2_2_ex1(capsys):
    excute_and_output('user_guide_3_2_2_ex1', capsys)

def test_user_guide_3_2_2_ex2(capsys):
    excute_and_output('user_guide_3_2_2_ex2', capsys)

def test_user_guide_3_3_1_ex1(capsys):
    excute_and_output('user_guide_3_3_1_ex1', capsys)

def test_user_guide_3_3_1_ex2(capsys):
    excute_and_output('user_guide_3_3_1_ex2', capsys)

def test_user_guide_3_3_2_ex1(capsys):
    excute_and_output('user_guide_3_3_2_ex1', capsys)

def test_user_guide_3_4_2_ex1(capsys):
    excute_and_output('user_guide_3_4_2_ex1', capsys)

def test_user_guide_3_4_3_ex1(capsys):
    excute_and_output('user_guide_3_4_3_ex1', capsys)

def test_user_guide_3_5_ex1(capsys):
    excute_and_output('user_guide_3_5_ex1', capsys)

def test_user_guide_3_5_ex2(capsys):
    excute_and_output('user_guide_3_5_ex2', capsys)

def test_user_guide_3_6_ex1(capsys):
    excute_and_output('user_guide_3_6_ex1', capsys)

def test_user_guide_3_7_1_ex1(capsys):
    excute_and_output('user_guide_3_7_1_ex1', capsys)

def test_user_guide_3_7_2_ex1(capsys):
    excute_and_output('user_guide_3_7_2_ex1', capsys)

def test_user_guide_3_8_1_ex1(capsys):
    excute_and_output('user_guide_3_8_1_ex1', capsys)

def test_user_guide_3_8_1_ex2(capsys):
    excute_and_output('user_guide_3_8_1_ex2', capsys)

def test_user_guide_3_8_2_ex1_2_3(capsys):
    test_names = ['user_guide_3_8_2_ex1', \
                 'user_guide_3_8_2_ex2', \
                 'user_guide_3_8_2_ex3']
    excute_all(test_names, capsys)

def test_user_guide_3_8_3_ex1_2(capsys):
    test_names = ['user_guide_3_8_3_ex1', \
                 'user_guide_3_8_3_ex2']
    excute_all(test_names, capsys)

def test_user_guide_4_1_ex1(capsys):
    excute_and_output('user_guide_4_1_ex1', capsys)

def test_user_guide_4_1_1_ex1(capsys):
    excute_and_output('user_guide_4_1_1_ex1', capsys)

@needGPUTest
def test_user_guide_4_1_2_ex1(capsys):
    excute_and_output('user_guide_4_1_2_ex1', capsys)

def test_user_guide_4_1_3_ex1(capsys):
    excute_and_output('user_guide_4_1_3_ex1', capsys)

def test_user_guide_4_1_3_ex2(capsys):
    excute_and_output('user_guide_4_1_3_ex2', capsys)

def test_user_guide_4_2_1_ex1(capsys):
    excute_and_output('user_guide_4_2_1_ex1', capsys)

def test_user_guide_4_3_1_ex1(capsys):
    excute_and_output('user_guide_4_3_1_ex1', capsys)

def test_user_guide_4_3_2_ex1(capsys):
    excute_and_output('user_guide_4_3_2_ex1', capsys)

def test_user_guide_4_5_1_ex1(capsys):
    excute_and_output('user_guide_4_5_1_ex1', capsys)

def test_user_guide_4_5_2_ex1(capsys):
    excute_and_output('user_guide_4_5_2_ex1', capsys)

def test_user_guide_4_5_3_ex1(capsys):
    excute_and_output('user_guide_4_5_3_ex1', capsys)

def test_user_guide_7_1(capsys):
    test_names = ['user_guide_7_1_ex1', \
                 'user_guide_7_1_1_ex1', \
                 'user_guide_7_1_1_ex2', \
                 'user_guide_7_1_2_ex1', \
                 'user_guide_7_1_2_ex2', \
                 'user_guide_7_1_3_ex1']
    excute_all(test_names, capsys)

def test_user_guide_7_2_1_ex1(capsys):
    excute_and_output('user_guide_7_2_1_ex1', capsys)

def test_user_guide_7_2_1_ex2(capsys):
    test_names = ['user_guide_7_2_1_ex2', \
                 'user_guide_7_2_1_ex3']
    excute_all(test_names, capsys)

def test_user_guide_7_2_2_ex1(capsys):
    excute_and_output('user_guide_7_2_2_ex1', capsys)

def test_user_guide_7_3_ex1(capsys):
    test_names = ['user_guide_7_3_ex1', \
                 'user_guide_7_3_1_ex1']
    excute_all(test_names, capsys)

def test_user_guide_7_4_ex1(capsys):
    excute_and_output('user_guide_7_4_ex1', capsys)

def test_user_guide_7_4_ex2(capsys):
    excute_and_output('user_guide_7_4_ex2', capsys)

def test_user_guide_7_4_1_ex1(capsys):
    excute_and_output('user_guide_7_4_1_ex1', capsys)

def test_user_guide_7_4_2_3(capsys):
    test_names = ['user_guide_7_4_2_ex1', \
                 'user_guide_7_4_2_ex2', \
                 'user_guide_7_4_3_ex1', \
                 'user_guide_7_4_3_ex2']
    excute_all(test_names, capsys)

def test_user_guide_7_5_1_ex1_2(capsys):
    with pytest.raises(Exception) as e_info:
        test_names = ['user_guide_7_5_1_ex1', \
                     'user_guide_7_5_1_ex2']
        excute_all(test_names, capsys)
    print(str(e_info.value))
    test_name = 'user_guide_7_5_1_ex2'
    exec(get_redirect_code(test_name))

def test_user_guide_7_6_ex1_2(capsys):
    test_names = ['user_guide_7_6_ex1', \
                 'user_guide_7_6_ex2']
    excute_all(test_names, capsys)

def test_user_guide_7_7_1_ex1(capsys):
    excute_and_output('user_guide_7_7_1_ex1', capsys)

def test_user_guide_7_7_2_3(capsys):
    test_names = ['user_guide_7_7_2_ex1', \
                 'user_guide_7_7_2_ex2', \
                 'user_guide_7_7_2_ex3', \
                 'user_guide_7_7_2_ex4', \
                 'user_guide_7_7_3_ex1', \
                 'user_guide_7_7_3_ex2']
    excute_all(test_names, capsys)

def test_user_guide_7_8_1_ex1(capsys):
    excute_and_output('user_guide_7_8_1_ex1', capsys)

def test_user_guide_7_8_1_ex2(capsys):
    excute_and_output('user_guide_7_8_1_ex2', capsys)

def test_user_guide_7_8_2(capsys):
    with pytest.raises(Exception) as e_info:
        test_names = ['user_guide_7_8_2_ex1', \
                     'user_guide_7_8_2_ex2', \
                     'user_guide_7_8_2_ex3']
        excute_all(test_names, capsys)
    print(str(e_info.value))
    test_name = 'user_guide_7_8_2_ex3'
    exec(get_redirect_code(test_name))

def test_user_guide_7_8_2_ex4(capsys):
    test_names = ['user_guide_7_8_2_ex1', \
                 'user_guide_7_8_2_ex4']
    excute_all(test_names, capsys)

def test_user_guide_7_9_1_ex1(capsys):
    excute_and_output('user_guide_7_9_1_ex1', capsys)

def test_user_guide_7_9_2_ex1(capsys):
    excute_and_output('user_guide_7_9_2_ex1', capsys)

def test_user_guide_7_9_3_ex1(capsys):
    excute_and_output('user_guide_7_9_3_ex1', capsys)

def test_user_guide_7_9_5(capsys):
    test_names = ['user_guide_7_9_5_ex1', \
                 'user_guide_7_9_5_ex2']
    excute_all(test_names, capsys)

def test_user_guide_7_10_1_ex1(capsys):
    excute_and_output('user_guide_7_10_1_ex1', capsys)
    time.sleep(0.01) # wait to save field, then next test can load

def test_user_guide_7_10_2_ex1(capsys):
    excute_and_output('user_guide_7_10_2_ex1', capsys)

def test_user_guide_8_1_2_ex1_2(capsys):
    test_names = ['user_guide_8_1_2_ex1', \
                 'user_guide_8_1_2_ex2']
    excute_all(test_names, capsys)

def test_user_guide_8_1_3_ex1(capsys):
    excute_and_output('user_guide_8_1_3_ex1', capsys)

def test_user_guide_8_1_4(capsys):
    test_names = ['user_guide_8_1_4_ex1', \
                 'user_guide_8_1_4_ex2', \
                 'user_guide_8_1_4_ex3']
    excute_all(test_names, capsys)

def test_user_guide_8_2_1_ex1(capsys):
    excute_and_output('user_guide_8_2_1_ex1', capsys)

def test_user_guide_8_2_1_ex2(capsys):
    excute_and_output('user_guide_8_2_1_ex2', capsys)

def test_user_guide_8_2_2_ex1(capsys):
    excute_and_output('user_guide_8_2_2_ex1', capsys)

def test_user_guide_8_3_ex1(capsys):
    excute_and_output('user_guide_8_3_ex1', capsys)

def test_user_guide_10_1_ex1(capsys):
    excute_and_output('user_guide_10_1_ex1', capsys)

def test_user_guide_10_1_1_ex1(capsys):
    excute_and_output('user_guide_10_1_1_ex1', capsys)

def test_user_guide_10_1_1_ex2_3(capsys):
    test_names = ['user_guide_10_1_1_ex2', \
                 'user_guide_10_1_1_ex3']
    excute_all(test_names, capsys)

def test_user_guide_10_1_2_ex1(capsys):
    excute_and_output('user_guide_10_1_2_ex1', capsys)

def test_user_guide_10_1_3_ex1(capsys):
    excute_and_output('user_guide_10_1_3_ex1', capsys)

def test_user_guide_10_2_ex1(capsys):
    excute_and_output('user_guide_10_2_ex1', capsys)
