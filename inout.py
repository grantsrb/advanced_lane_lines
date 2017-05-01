import os

def read_paths(dir_path, file_exts={'jpg':True,'png':True}):
    path_list = []
    for dir_name, subdir, file_list in os.walk(dir_path):
        for f in file_list:
            f_ext = f.split('.')[-1]
            if f_ext.lower() in file_exts:
                path_list.append(os.path.join(dir_name,f))
    return path_list
