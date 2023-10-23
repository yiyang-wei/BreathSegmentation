import pandas as pd


class BreathParamTable:

    def __init__(self, in_file_path):
        self.in_file_path = in_file_path
        self.df = pd.read_csv(in_file_path, header=0, index_col=0)
        self.start_number = self.df.index[0]
        self.end_number = self.df.index[-1]

    def join(self, breath_labels):
        # join the breath labels df to the table by breath number
        self.df = self.df.join(breath_labels, how='left', on='breath_number')

    def save(self):
        self.df.to_csv(self.in_file_path)

    def save_as(self, out_file_path):
        self.df.to_csv(out_file_path)


if __name__ == '__main__':
    import os
    FOLDER = r"C:\Users\weiyi\Downloads\original_parameter_table\original_parameter_table"
    FILE_NAME = "EVLP818_param_table.csv"
    FILE_PATH = os.path.join(FOLDER, FILE_NAME)
    table = BreathParamTable(FILE_PATH)
    print(table.df.columns)
    print(table.df.head())
    print(table.df.tail())
    print(table.df.shape)