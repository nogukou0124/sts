import file
import pre_pro

learns = file.pre_learn_data()
tests = file.pre_test_data()

new_learns = pre_pro.pre_processing(learns)
new_tests = pre_pro.pre_processing(tests)
print(new_tests)