import os
import ctypes


def deleteAll(path:str):
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))  
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))
            
def displayFilesRec(path:str):
    for root, dirs, files in os.walk(path):
        print(root)
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))
            
def unicDirsRec(path:str):
    lst = set()
    for _, dirs, _ in os.walk(path):
        for dir in dirs:
            lst.add(dir)
    return lst

def countFiles(directory:str):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])


        
    