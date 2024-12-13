import os, sys

path = './ab_file/'

dirs = sorted(os.listdir(path))

for d in dirs[1000:2000]:
    if d == '.DS_Store': continue
    if os.path.exists(path + d +
                      '/cpg_a.txt') and os.path.exists(path + d +
                                                       '/cpg_b.txt'):
        continue

    os.system(
        'cd ./joern; ./joern --script ../locateFunc.sc --params inputFile=.' +
        path + d + '/a/,outFile=.' + path + d + '/cpg_a.txt')

    os.system(
        'cd ./joern; ./joern --script ../locateFunc.sc --params inputFile=.' +
        path + d + '/b/,outFile=.' + path + d + '/cpg_b.txt')

    os.system('python3 locate_and_align.py ' + path + d + '/')
