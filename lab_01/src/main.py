import os

import config as config
from solver import solve_equation
from file_reader import load_tables
from draw import plot_results

def main():

    params = config.build_params()
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data")
    tables = load_tables(data_dir)
    results = solve_equation(params, tables)
    plot_results(results)

if __name__ == "__main__":
    main()

