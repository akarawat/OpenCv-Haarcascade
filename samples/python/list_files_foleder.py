import os

# folder path
dir_path = 'pictures'

# list to store files
res = []
iCount = 0
# Iterate directory
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        res.append(path)
        #strFname = {iCount}
        #strFname = "pictures/"+ strFname.format(*res)
        #concatenated_value = ' posted a challenge to '.join(strFname.format(*res))
        #print(concatenated_value)
        #print(strFname.format(*res))
        #name_strings = ['Team 1', 'Team 2']
        print("% posted a challenge to %s", (res))
        iCount+=1

print(res[2])

# name_strings = ['Team 1', 'Team 2']
# print("%s posted a challenge to %s" % tuple(name_strings))

# array_of_strings = ['Team1', 'Team2']
# message = '{0}'
# print(message.format(*array_of_strings))
