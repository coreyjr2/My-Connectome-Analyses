import os
import gzip
import shutil
import tarfile

path1 = 'S:\\MyConnectome\\RNASeq\\'
'''
for file in os.listdir(path1):
    with gzip.open(path1+file, 'rb') as f_in:
        with open(path1 + file[:-3], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

'''

path2 = 'S:\\MyConnectome\\manual_downloads\\'
for file in os.listdir(path2):
    if '.tgz' in file:
        tar = tarfile.open(path2+file)
        tar.extractall(path=path2)
        tar.close()
