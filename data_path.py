f = open("digits/svhn_list.txt", "r")
my_path = "/work/xuweinan/Generate_To_Adapt_modified/digits"
r = f.readlines()
for i in range(0, len(r)):
    print(r[i])
    pre_path, rest = r[i].split('digits', 1)
    r[i] = my_path + rest
    print("new: " + r[i])
    f = open("digits/server_svhn_list.txt", "a+")
    f.write(r[i])
    f.close()
