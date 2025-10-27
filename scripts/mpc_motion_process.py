import csv
import numpy as np

# 判断字符串是否为有效数字
def is_number(value):
    try:
        float(value)  # 尝试转换为浮动数
        return True
    except ValueError:
        return False

def convert_csv_to_txt(csv_file, output_txt_file, selected_columns):
    """
    将CSV文件转换为TXT格式,根据指定的列标题来选择性地存储数据

    :param csv_file: 输入的CSV文件路径
    :param output_txt_file: 输出的TXT文件路径
    :param selected_columns: 一个列表，包含要存储的列标题
    """
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        
        # 读取标题行
        header = next(csv_reader)

        # 用于存储选中的列的索引
        selected_indices = []
        missing_columns = []  # 用于存储未找到的列名
        
        for col in selected_columns:
            try:
                # 找出需要的列的索引
                selected_indices.append(header.index(col))
            except ValueError:
                # 如果找不到标题，加入 missing_columns 列表
                missing_columns.append(col)

        # 如果有未找到的标题，打印出来
        if missing_columns:
            print("未找到以下标题名：", missing_columns)
        else:
            print("所有列标题均已找到。", selected_columns)

        # 存储CSV数据
        motion_data = []

        row_count = 0
        
        # 遍历 CSV 中的每一行数据
        for row in csv_reader:
            # 每10行取一行
            if row_count % 8 == 0:
                selected_data = []
                for idx in selected_indices:
                    value = row[idx].strip()
                    try:
                        if value == '' or not is_number(value):
                            print(f"Invalid value at row {row_count}, column '{header[idx]}': '{row[idx]}'")
                            selected_data.append(0)
                        else:
                            selected_data.append(float(value))
                    except ValueError:
                        selected_data.append(0)
                motion_data.append(selected_data)
            
            # 更新行计数器
            row_count += 1
        
    
    # 将数据保存为txt文件
    with open(output_txt_file, 'w') as txt_file:
        for i, frame in enumerate(motion_data):
            # 将每一行数据转换为字符串并写入txt文件，格式为[elem1, elem2, ...]，并在行与行之间加上逗号
            if i != len(motion_data) - 1:  # 如果不是最后一行，添加逗号
                txt_file.write(f"[{', '.join(map(str, frame))}],\n")
            else:  # 如果是最后一行，不加逗号
                txt_file.write(f"[{', '.join(map(str, frame))}]\n")
    
    print(f"数据已成功保存为 {output_txt_file}")

# 使用示例
file_name_list = ["forward_v2", "left_v2", "right_v2", "back_v2", "left_y", "right_y"]
for i in file_name_list:
    csv_file_root = '/home/liuyun/MPC_data_go2/'
    csv_file = csv_file_root + i + '.csv'  # CSV文件路径
    output_txt_file = i + '.txt'  # 输出的TXT文件路径''
    selected_columns = ['/debug/qb/meas[0]', '/debug/qb/meas[1]', '/debug/qb/meas[2]', # base pos 3
                        '/debug/qb/meas[6]','/debug/qb/meas[7]', '/debug/qb/meas[8]', '/debug/qb/meas[9]', # base ori 4
                        '/debug/qj/meas[0]', '/debug/qj/meas[1]', '/debug/qj/meas[2]', '/debug/qj/meas[3]', # joint pos 12
                        '/debug/qj/meas[4]', '/debug/qj/meas[5]', '/debug/qj/meas[6]', '/debug/qj/meas[7]',
                        '/debug/qj/meas[8]', '/debug/qj/meas[9]', '/debug/qj/meas[10]', '/debug/qj/meas[11]',
                        '/debug/pf/meas[0]', '/debug/pf/meas[1]', '/debug/pf/meas[2]', '/debug/pf/meas[3]', # foot pos 12
                        '/debug/pf/meas[4]', '/debug/pf/meas[5]', '/debug/pf/meas[6]', '/debug/pf/meas[7]',
                        '/debug/pf/meas[8]', '/debug/pf/meas[9]', '/debug/pf/meas[10]', '/debug/pf/meas[11]',
                        '/debug/vb/meas[0]', '/debug/vb/meas[1]', '/debug/vb/meas[2]',  # base linear vel 3
                        '/debug/vb/meas[5]', '/debug/vb/meas[4]', '/debug/vb/meas[3]', # base angular vel 3， 注意顺序，原始数据是zyx
                        '/debug/vj/meas[0]', '/debug/vj/meas[1]', '/debug/vj/meas[2]', '/debug/vj/meas[3]', # joint vel 12
                        '/debug/vj/meas[4]', '/debug/vj/meas[5]', '/debug/vj/meas[6]', '/debug/vj/meas[7]',
                        '/debug/vj/meas[8]', '/debug/vj/meas[9]', '/debug/vj/meas[10]', '/debug/vj/meas[11]',
                        '/debug/vf/meas[0]', '/debug/vf/meas[1]', '/debug/vf/meas[2]', '/debug/vf/meas[3]', # foot vel 12
                        '/debug/vf/meas[4]', '/debug/vf/meas[5]', '/debug/vf/meas[6]', '/debug/vf/meas[7]',
                        '/debug/vf/meas[8]', '/debug/vf/meas[9]', '/debug/vf/meas[10]', '/debug/vf/meas[11]',
                        ]# 要选择存储的列标题列表
    convert_csv_to_txt(csv_file, output_txt_file, selected_columns)