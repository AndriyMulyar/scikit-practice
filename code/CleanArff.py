import fileinput
import re



def removeUnneccessaryTags(filename):
    for line in fileinput.FileInput(filename, inplace=1):
        if('@inputs' in line or '@outputs' in line):
            print("".rstrip());
        else:
            print(line.rstrip())



for i in range(1,6):
    removeUnneccessaryTags("../datasets/ecoli3-8.6/ecoli3-5-%stra.dat" % i)
    removeUnneccessaryTags("../datasets/ecoli3-8.6/ecoli3-5-%stst.dat" % i)