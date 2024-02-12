import logging
from pathlib import Path
import re
import pandas as pd
import yaml

def parse_config(config_file):

    with open(config_file, "rb") as f:
        config = yaml.safe_load(f)
    return config


def cleaning_data(col):    
     col = col.map(lambda x: re.sub(r"https?://\S+|www\.\S+","",x) )
     col = col.map(lambda x: re.sub("["
                                                                                   u"\U0001F600-\U0001F64F"
                                                                                   u"\U0001F300-\U0001F5FF"
                                                                                   u"\U0001F680-\U0001F6FF"
                                                                                   u"\U0001F1E0-\U0001F1FF"
                                                                                   u"\U00002702-\U000027B0"
                                                                                   u"\U000024C2-\U0001F251"
                                                                                   "]+","", x, flags=re.UNICODE))
     
     col = col.map(lambda x: re.sub(r"[^a-zA-Z0-9\s\"\',:;?!.()]", " ",x))     
     col = col.map(lambda x: re.sub(r"\s\s+", " ",x))         
     col = col.map(lambda x: re.sub(r"^\"", "",x))                                                                                                                                                                                                                                                                                       "]+","", x, flags=re.UNICODE))
