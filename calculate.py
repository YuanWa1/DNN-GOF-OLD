import pandas as pd
import numpy as np 
import sys
import os

def check(column_name,df):
    df[column_name + '_check'] = np.where(df[column_name] < 0.05, 'TRUE', 'FALSE')

def count(column_name,df,final_count):
    true_count = (df[column_name + '_check'] == 'TRUE').sum()
    final_count[column_name] = true_count*0.001

def main():
    print(f"Processing file: {sys.argv[1]}")
    #skip_rows = list(range(0,2)) + list(range(4,1004))
    df = pd.read_csv(sys.argv[1],skiprows= 0)
    
    df[['P_value_shallow_0', 'P_value_shallow_1']] = df['P_value_shallow'].str.split(' ', expand=True)
    df[['P_value_dnn1_0', 'P_value_dnn1_1']] = df['P_value_dnn1'].str.split(' ', expand=True)
    df[['P_value_dnn2_0', 'P_value_dnn2_1']] = df['P_value_dnn2'].str.split(' ', expand=True)
    
    df[['P_value_shallow_0', 'P_value_shallow_1', 'P_value_dnn1_0', 'P_value_dnn1_1', 'P_value_dnn2_0', 'P_value_dnn2_1']] \
        = df[['P_value_shallow_0', 'P_value_shallow_1', 'P_value_dnn1_0', 'P_value_dnn1_1', 'P_value_dnn2_0', 'P_value_dnn2_1']].astype(float)
    
    df.drop(['P_value_shallow', 'P_value_dnn1', 'P_value_dnn2'], axis=1, inplace=True)
    
    # new_filename = sys.argv[1].split('.')[0] + '_updated.csv'
    # df.to_csv(new_filename, index=False)
    
    column_names = df.columns
    
    final_count = {}
    for column in column_names:
        check(column,df=df)

    for column in column_names:
        count(column,df=df,final_count=final_count)
    
    if not os.path.exists('final_result.txt'):
        f = open("final_result.txt",'x')
    f = open("final_result.txt",'a')
    f.write(sys.argv[1] + ':\n')
    f.write(str(final_count)+'\n\n')
    f.close()
    
    
if __name__ == "__main__":
    main()
