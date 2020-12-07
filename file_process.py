import os 


def traverse_file(dir_path, output_dir="", process_functions=None, save_function=None):
    if os.path.isfile(dir_path):
        print(dir_path)
        if process_functions is not None:
            process_functions(dir_path)
        if save_function is not None:
            file_name = os.path.basename(dir_path)
            save_function(dir_path, os.path.join(output_dir, file_name))
        return True
    success_num = 0  # 转换成功的文件数
    file_list = []
    dir_list = []
    for file_name in os.listdir(dir_path):
        if file_name.startswith("."):
            continue
        file_path = os.path.join(dir_path, file_name)
        if os.path.isfile(file_path):
            file_list.append(file_path)
        elif os.path.isdir(file_path):
            dir_list.append(file_path)
        else:
            print("暂时不支持该路径：{}".format(file_path))

    for file_path in sorted(file_list):
        # print(file_path)
        if process_functions is not None:
            process_functions(file_path)
        if save_function is not None:
            file_name = os.path.basename(file_path)
            save_function(file_path, os.path.join(output_dir, file_name))
        success_num += 1
        
    for dir_path in sorted(dir_list):
        print("dir_path:", dir_path)
        file_name = os.path.basename(dir_path)
        result = traverse_file(dir_path, os.path.join(output_dir, file_name),  process_functions, save_function)
        success_num += result
    return success_num
