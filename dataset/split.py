import pandas as pd
from sklearn.model_selection import train_test_split
import os
import argparse

parser=argparse.ArgumentParser(description="Drug-Drug Interaction split create. It creates 5 datasets with a 60-20-20 split.")
parser.add_argument("--drugbank-csv", dest="drugbank", help="Path to the drugbank csv dataset.")

args=parser.parse_args()

dataset=args.drugbank
#print(dataset)
if dataset is None:
    raise ValueError("No dataset has been passed. Please use the --help command to understand how to use split.py.")

for i in range(1,6):
    train_dataframe=[]
    val_dataframe=[]
    test_dataframe=[]

    for val in dataset['Y'].unique():
        subset=dataset[dataset['Y']==val]

        tmp,test=train_test_split(subset,test_size=0.2)
        train,val=train_test_split(tmp,test_size=0.25)

        train_dataframe.append(train)
        val_dataframe.append(val)
        test_dataframe.append(test)

    train_final=pd.concat(train_dataframe)
    validation_final=pd.concat(val_dataframe)
    test_final=pd.concat(test_dataframe)

    if not os.path.exists(f"{os.path.dirname(os.path.realpath(__file__))}{os.sep}Dataset_{i}"):
        os.makedirs(f"{os.path.dirname(os.path.realpath(__file__))}{os.sep}Dataset_{i}")
    if not os.path.exists(f"{os.path.dirname(os.path.realpath(__file__))}{os.sep}Dataset_{i}{os.sep}train"):
        os.makedirs(f"{os.path.dirname(os.path.realpath(__file__))}{os.sep}Dataset_{i}{os.sep}train")
    if not os.path.exists(f"{os.path.dirname(os.path.realpath(__file__))}{os.sep}Dataset_{i}{os.sep}validation"):
        os.makedirs(f"{os.path.dirname(os.path.realpath(__file__))}{os.sep}Dataset_{i}{os.sep}validation")
    if not os.path.exists(f"{os.path.dirname(os.path.realpath(__file__))}{os.sep}Dataset_{i}{os.sep}test"):
        os.makedirs(f"{os.path.dirname(os.path.realpath(__file__))}{os.sep}Dataset_{i}{os.sep}test")
    
    train_final.to_csv(f"{os.path.dirname(os.path.realpath(__file__))}{os.sep}Dataset_{i}{os.sep}train{os.sep}train.csv",sep=";")
    validation_final.to_csv(f"{os.path.dirname(os.path.realpath(__file__))}{os.sep}Dataset_{i}{os.sep}validation{os.sep}val.csv",sep=";")
    test_final.to_csv(f"{os.path.dirname(os.path.realpath(__file__))}{os.sep}Dataset_{i}{os.sep}test{os.sep}test.csv",sep=";")
