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

    df = pd.read_csv(sys.argv[1],skiprows=2)

    column_names = df.columns
    final_count = {}
    for column in column_names:
        check(column,df=df)

    for column in column_names:
        count(column,df=df,final_count=final_count)
    
    # df = df._append(final_count, ignore_index=True)
    # print(df)
    # loc = df.columns.get_loc('p_lm_check')
    # df.insert(loc, '', np.nan)
    # new_filename = sys.argv[1].split('.')[0] + '_updated.csv'
    # df.to_csv(new_filename, index=False, na_rep='')
    if not os.path.exists('final_result.txt'):
        f = open("final_result.txt",'x')
    f = open("final_result.txt",'a')
    f.write(sys.argv[1] + ':\n')
    f.write(str(final_count)+'\n\n')
    f.close()
    
    
if __name__ == "__main__":
    main()
