f = open("digits/mnist_list.txt", "r")
my_path = "/work/xuweinan/Generate_To_Adapt_modified/Generate_To_Adapt/data/digits"
r = f.readlines()
for i in range(0, len(r)):
    print(r[i])
    pre_path, rest = r[i].split('digits', 1)
    r[i] = my_path + rest
    print("new: " + r[i])
    f = open("digits/tmp_server_mnist_list.txt", "a+")
    f.write(r[i])
    f.close()
