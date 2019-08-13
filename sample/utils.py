def data_loader(f_name, l_name):
    with open(f_name, mode='r', encoding='utf-8') as f:
        data = list(set(f.readlines()))
        label = [l_name for i in range(len(data))]

        return data, label
