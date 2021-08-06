from data_parser import xml_to_csv, drop_invalid_lines, preprocess_data, preprocess_demo
from train import demo_test

# input_file should be the demo data file
input_file = 'mps.dataset.test.xlsx'
output_file = 'test.csv'

if __name__ == '__main__':
    xml_to_csv(input_filename=input_file, output_filename=output_file)
    df1 = preprocess_demo(output_file)
    demo_test(df1)
