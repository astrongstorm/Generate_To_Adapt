import os

count = 0
for subdir, dirs, files in os.walk('/work/xuweinan/Visda_Data'):
    for file in files:
        # print(subdir)
        # if count == 0:
        print("subdir: {}".format(subdir))
        print("subdir.split('/')[-1]: {}".format(subdir.split('/')[-1]))
        label = subdir.split('/')[-1]
        if label.isdigit() and file != '.DS_Store':
            filepath = subdir + os.sep + file
            full_path = os.getcwd() + os.sep + filepath + " " + str(label) + "\n"
            f = open("digits/usps_list.txt", "a+")
            f.write(full_path)
            f.close()
            print("full path: {}".format(full_path))
            count += 1
    print(count)