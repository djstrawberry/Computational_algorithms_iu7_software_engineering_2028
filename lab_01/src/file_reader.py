import os

def load_tables(data_dir: str) -> dict:

    tables = {}
    
    tables["I_t"] = load_1D_table(os.path.join(data_dir, "I_t.txt"))
    tables["Nh"] = load_2D_table(os.path.join(data_dir, "Nh.txt"))
    tables["sigma"] = load_2D_table(os.path.join(data_dir, "sigma.txt"))
    tables["c"] = load_2D_table(os.path.join(data_dir, "c.txt"))
    tables["q"] = load_2D_table(os.path.join(data_dir, "q.txt"))

    return tables

def load_1D_table(file_path: str) -> dict:

    x_args = []
    y_args = []

    with open (file_path, 'r') as file:
        for line in file:
            columns = line.split()
            x_args.append(float(columns[0]))
            y_args.append(float(columns[1]))

    return {"x": x_args, "y": y_args} 

def load_2D_table(file_path: str) -> dict:

    T_args = []
    values = []

    with open (file_path, 'r') as file:

        header = file.readline().split()
        p_args = [float(p) for p in header[1:]]

        for line in file:
            columns = line.split()
            T_args.append(float(columns[0]))
            values.append([float(value) for value in columns[1:]]) 

    return {"T": T_args, "p": p_args, "values": values} 