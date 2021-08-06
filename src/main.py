from data_parser import xml_to_csv, drop_invalid_lines, preprocess_data
from train import train

input_file = 'mps.dataset.xlsx'
output_file = 'test.csv'

if __name__ == '__main__':
    xml_to_csv(input_filename=input_file, output_filename=output_file)
    df = drop_invalid_lines()
    df1 = preprocess_data(df)
    train(df1)
