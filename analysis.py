from matplotlib import pyplot as plt
import file

def view_analysis():
    data_list = []
    with open("./db/test/STS.output.images.txt", 'r', encoding='utf-8') as file:
        for line in file.read().splitlines():
            data_list.append(line)

    correct_data = []
    with open("./db/test/STS.gs.images.txt", 'r', encoding='utf-8') as file:
        for line in file.read().splitlines():
            correct_data.append(line)
            
    docs = []
    with open("./db/test/STS.input.images.txt", 'r', encoding='utf-8') as file:
        for line in file.read().splitlines():
            docs.append(line)

    fig = plt.figure()

    x = []
    y = []
    constant_max = 30
    constant_min = 1

    for (data,correct) in zip(data_list,correct_data):
        x.append(float(data))
        y.append(float(correct))

    
    file_path = "./db/analysis/data.txt"
    with open(file_path, 'w', encoding='utf-8') as file:
        for (data,correct,doc) in zip(x,y,docs):
            file.write(f"[{data},{correct}] {doc}\n")

    plt.scatter(x,y)
    fig.savefig("./images/img.jpg")
    


        

