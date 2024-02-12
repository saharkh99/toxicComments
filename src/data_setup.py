from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader, Dataset
import numpy as np

class comDataset(Dataset):
     def __init__(self, Comments_, Labels_, tokenizer):
        self.comments = Comments_.copy()
        self.labels = Labels_.copy()
        col = col.map(lambda x: tokenizer(x, padding="max_length", truncation=True, return_tensors="pt"))
     def __len__(self):
        return len(self.labels)

     def __getitem__(self, idx):
        comment = self.comments.loc[idx ,"comment_text"]
        label = np.array(self.labels.loc[idx,:])
        return comment, label
    
def etl(col):
    tockenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    X_train, X_test, Y_train, Y_test = train_test_split(col.iloc[:,1],col.iloc[:,2:], test_size=0.1)
    train_data = comDataset(X_train,Y_train, tockenizer)
    train_dataloader = DataLoader(train_data, batch_size = 32, shuffle = True) 
    test_data = comDataset(X_test,Y_test, tockenizer)
    test_dataloader = DataLoader(test_data, batch_size = 32, shuffle = True)
    return train_dataloader, test_dataloader

