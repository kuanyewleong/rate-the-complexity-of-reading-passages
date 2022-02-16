import transformers
from transformers import RobertaTokenizer, RobertaModel

import pandas as pd 
import numpy as np
import torch
import torch.nn as nn

import re
import string
import nltk
from nltk.corpus import stopwords

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def remove_stopwords(text):
    # nltk.download('stopwords')
    stop_words = stopwords.words('english')
    more_stopwords = ['u', 'im', 'c']
    stop_words = stop_words + more_stopwords
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    return text


def stemm_text(text):
    stemmer = nltk.SnowballStemmer("english")
    text = ' '.join(stemmer.stem(word) for word in text.split(' '))
    return text


class BERT_Arch(nn.Module):

    def __init__(self, bert):
      
        super(BERT_Arch, self).__init__()

        self.bert = bert 
        
        # dropout layer
        self.dropout = nn.Dropout(0.7)
        
        # relu activation function
        self.relu =  nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # dense layer 1
        # self.fc1 = nn.Linear(768,512)
        self.fc1 = nn.Linear(1024,1)
        
        # dense layer 2 (Output layer)
        # self.fc2 = nn.Linear(512,1)


    #define the forward pass
    def forward(self, sent_id, mask):

        #pass the inputs to the model  
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        
        x = self.fc1(cls_hs)

        #x = self.relu(x)

        #x = self.dropout(x)
        
        #x = self.fc2(x)

        return x


test_df = pd.read_csv('../input/commonlitreadabilityprize/test.csv')

# data cleaning
test_df['excerpt_clean'] = test_df['excerpt'].apply(clean_text)

# stopwords removing
test_df['excerpt_clean'] = test_df['excerpt_clean'].apply(remove_stopwords)

# Stemming/ Lematization
test_df['excerpt_clean'] = test_df['excerpt_clean'].apply(stemm_text)

test_text = test_df['excerpt_clean']

# Load the roberta-base tokenizer
tokenizer = RobertaTokenizer.from_pretrained('../input/huggingface-roberta-variants/roberta-large/roberta-large')

# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = 100,
    padding='max_length',
    truncation=True
)

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])

use_cuda: bool = torch.cuda.is_available()
device_string = "cuda" if use_cuda else "cpu"
device = torch.device(device_string)
# import roberta-base pretrained model
bert = RobertaModel.from_pretrained('../input/huggingface-roberta-variants/roberta-large/roberta-large')
# create model
model = BERT_Arch(bert)
# model = torch.nn.DataParallel(model, device_ids=[0])
model = model.to(device)
model.load_state_dict(torch.load("../input/finetune-18to24-layers/epoch_20_18to24_1fc_saved_weights.pt", map_location=device_string))
model.eval()

with torch.no_grad():
    preds = model(test_seq.to(device), test_mask.to(device))
    preds = preds.detach().cpu().numpy()

print("preds: ", preds)
# make prediction and save to file 
submission_file_path = "../input/commonlitreadabilityprize/sample_submission.csv"
sample_submission = pd.read_csv(submission_file_path)
sample_submission["target"] = preds
sample_submission.to_csv("./submission.csv", index=False)
