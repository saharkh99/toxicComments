import argparse
import pandas as pd
import os
from src import utility
from transformers import DistilBertForSequenceClassification
from transformers import DistilBertForSequenceClassification
from torch.nn import nn
from src import train 
from src import data_setup
from src import test

def Main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder")
    parser.add_argument("--input_file_name")
    parser.add_argument("--output_folder")
    args = parser.parse_args()
    input_folder = args.input_folder
    input_file_name = args.input_file_name
    output_folder = args.output_folder
    print(f"input_folder is {input_folder}")
    df = pd.read_csv(os.path.join(input_folder, input_file_name))
    df.to_csv(os.path.join(output_folder,input_file_name))
    pre_df = utility.cleaning_data(df['comment'])
    Distil_bert = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    Distil_bert.classifier = nn.Sequential(
                    nn.Linear(768,7),
                    nn.Sigmoid()
                  )
    trian_loader, test_loader = data_setup(pre_df)
    train.train_Model(Distil_bert ,trian_loader,0.01, 30)
    test.evaluate_Model(Distil_bert, test_loader)

    
if(__name__ == "__main__"):
    Main()