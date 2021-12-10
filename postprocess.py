from sys import argv
from re import sub
from glob import glob

def main(files):
    for filepath in files:
        for file in glob(filepath):
            with open(file, 'r') as f:
                data=f.readlines()
            data = data [1::3]
            for i in range(len(data)):
                data[i] = sub('<[^>]*>','',data[i]).strip(' ')
            
            with open(file+'.test.txt', 'w') as f:
                f.writelines(data)

if __name__ == '__main__':
    if len(argv)<=1:
        print('usage: postprocess file1 file2 ...')
    else:
        main(argv[1:])