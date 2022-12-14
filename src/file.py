def input_file(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file.read().splitlines():
          double_sentences = line.split('\t')
          data.append(double_sentences[0])
          data.append(double_sentences[1])
    return data

def gs_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file.read().splitlines():
          data.append(float(line) / 5)
    return data

         
def pre_learn_data():
    learns = []
    learn_data = []
    learn_data.append("./db/learn/STS.input.OnWN.txt")
    learn_data.append("./db/add/STS.input.MSRvid.txt")
    learn_data.append("./db/add/STS.input.headlines.txt")
    for learn in learn_data:
        for input in input_file(learn):
            learns.append(input)
    return learns

def pre_test_data():
    test_path = "./db/test/STS.input.images.txt"
    return input_file(test_path)

output_filepath = "./db/test/STS.output.images.txt"
def output_file(output_data):
    with open(output_filepath, 'w', encoding='utf-8') as file:
      	for output in output_data:
        	file.write(f'{output}\n')