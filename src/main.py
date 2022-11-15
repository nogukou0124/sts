import file
import pre_pro
import train_and_test
import analysis

# データの用意
learns = file.pre_learn_data()
tests = file.pre_test_data()

# 前処理
new_learns = pre_pro.pre_processing(learns)
new_tests = pre_pro.pre_processing(tests)
# new_tests = tests

# 訓練とテスト
output = train_and_test.train_and_test(new_learns,learns,new_tests)

# データの出力
file.output_file(output)

# データの分析
analysis.view_analysis()

