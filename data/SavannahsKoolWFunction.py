import torch
from torch import nn
from pathlib import Path
import pandas as pd
# File names of GloVe datasets represent dimensions - can change later if we want

"""
SO FAR JUST A RIP-OFF OF THE ACM AI RNN NOTEBOOK LOL
"""
# TODO: fix everything to fit our model


""" 
Change GloVe embeddings to fit our network
"""
# TODO: look at dimensions
with open('glove.6B.50d.txt','rt') as fi:
    full_content = fi.read() # read the file
    full_content = full_content.strip() # remove leading and trailing whitespace
    full_content = full_content.split('\n') # split the text into a list of lines

for i in range(len(full_content)):
    i_word = full_content[i].split(' ')[0] # get the word at the start of the line
    i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]] # get the embedding of the word in an array
    # add the word and the embedding to our lists
    vocab.append(i_word)
    embeddings.append(i_embeddings)

# convert our lists to numpy arrays:
import numpy as np
vocab_npa = np.array(vocab)
embs_npa = np.array(embeddings)


"""
Add tokens and embeddings to handle padding and unknowns
"""
# insert tokens for padding and unknown words into our vocab
vocab_npa = np.insert(vocab_npa, 0, '<pad>')
vocab_npa = np.insert(vocab_npa, 1, '<unk>')
print(vocab_npa[:10])

# make embeddings for these 2:
# -> for the '<pad>' token, we set it to all zeros
# -> for the '<unk>' token, we set it to the mean of all our other embeddings

pad_emb_npa = np.zeros((1, embs_npa.shape[1])) 
unk_emb_npa = np.mean(embs_npa, axis=0, keepdims=True) 

#insert embeddings for pad and unk tokens to embs_npa.
embs_npa = np.vstack((pad_emb_npa,unk_emb_npa,embs_npa))
print(embs_npa.shape)


"""
Give embedding layer dimensions
"""
# embedding layer should have dimensions = number of words in vocab × number of dimensions in embedding
my_embedding_layer = torch.nn.Embedding.from_pretrained(torch.from_numpy(embs_npa).float())


"""
Using our dataset
"""
DATA_PATH = 'train.csv'
if not Path(DATA_PATH).is_file():
    gdd.download_file_from_google_drive(
        file_id='1zfM5E6HvKIe7f3rEt1V2gBpw5QOSSKQz',
        dest_path=DATA_PATH,
    )

# load train.csv into a pandas dataframe
df = pd.read_csv("train.csv")
df.head()


"""
Class
"""
class LSTMDataset(torch.utils.data.Dataset):
    def __init__(self, df, vocab, max_seq_length, pad_token, unk_token):
        # make a list of our labels
        self.labels = df.label.tolist()

        # make a dictionary converting each word to its id in the vocab, as well
        # as the reverse lookup
        self.word2idx = {term:idx for idx,term in enumerate(vocab)}
        self.idx2word = {idx:word for word,idx in self.word2idx.items()} 
        
        self.pad_token,self.unk_token = pad_token,unk_token

        self.input_ids = [] 
        self.sequence_lens = [] 
        self.labels = []

        for i in range(df.shape[0]):
            # clean up each sentence and turn it into tensor containing the  
            # token ids of each word. Also add padding to make them all the 
            # same length as the longest sequence
            input_ids,sequence_len = self.convert_text_to_input_ids(
                df.iloc[i].review,
                pad_to_len = max_seq_length) 
            
            self.input_ids.append(input_ids.reshape(-1))
            self.sequence_lens.append(sequence_len)
            self.labels.append(df.iloc[i].label)
        
        #sanity checks
        assert len(self.input_ids) == df.shape[0]
        assert len(self.sequence_lens) == df.shape[0]
        assert len(self.labels) == df.shape[0]
    
    def convert_text_to_input_ids(self,text,pad_to_len):
        # truncate excess words (beyond the length we should pad to)
        words = text.strip().split()[:pad_to_len]

        # add padding till we've reached desired length 
        deficit = pad_to_len - len(words) 
        words.extend([self.pad_token]*deficit)

        # replace words with their id
        for i in range(len(words)):
            if words[i] not in self.word2idx:
                # if word is not in vocab, then use <unk> token
                words[i] = self.word2idx[self.unk_token] 
            else:
                # else find the id associated with the word 
                words[i] = self.word2idx[words[i]] 
        return torch.Tensor(words).long(),pad_to_len - deficit

    def __len__(self):
        # Make dataset compatible with len() function
        return len(self.input_ids)
    
    def __getitem__(self, i):
        # for the ith indexm return a dictionary containing id, length and label
        sample_dict = dict()
        sample_dict['input_ids'] = self.input_ids[i].reshape(-1)
        sample_dict['sequence_len'] = torch.tensor(self.sequence_lens[i]).long()
        sample_dict['labels'] = torch.tensor(self.labels[i]).type(torch.FloatTensor)
        return sample_dict
    
"""
Define ML model
"""
class LSTMEncoder(torch.nn.Module):
    def __init__(self, config):
        super(LSTMEncoder, self).__init__()
        
        # use the pretrained embeddings and check whether or not we should
        # freeze embeddings from our config dict
        pretrained_embeddings = config['pretrained_embeddings'] if 'pretrained_embeddings' in config else None
        freeze_embeddings = config['freeze_embeddings'] if 'freeze_embeddings' in config else False
        if pretrained_embeddings is not None:
            # use pretrained embeddings
            self.vocab_size = pretrained_embeddings.shape[0]
            self.embedding_dim = pretrained_embeddings.shape[1]
            self.embedding = torch.nn.Embedding.from_pretrained(
                torch.from_numpy(pretrained_embeddings).float(),
                freeze=freeze_embeddings
                )
        else:
            # use randomly initialized embeddings
            assert 'vocab' in config and 'embedding_dim' in config
            self.vocab_size = config['vocab'].shape[0]
            self.embedding_dim = config['embedding_dim']
            if freeze_embeddings:
                # why would you do this?
                print(
                    'WARNING:Freezing Randomly Initialized Embeddings!!😭😭😭'
                    )
            self.embedding = torch.nn.Embedding(
                self.vocab_size,
                self.embedding_dim,
                freeze = freeze_embeddings
                )
        
        # store some values from the config 
        self.hidden_size = config['hidden_size']
        self.lstm_unit_cnt = config['lstm_unit_cnt']

        # initialize LSTM 
        self.lstm = torch.nn.LSTM(
            input_size = self.embedding_dim,
            hidden_size = self.hidden_size,
            num_layers = self.lstm_unit_cnt,
            
            # batch_first = T -> input dim are [batch x sentence x embedding]
            # batch_first = F -> input dim are [sentence x batch x embedding]
            batch_first = True,

            # if bidirectional is true, then the seqeunce is passed through in 
            # both forward and backward directions and the results are 
            # concatenated. Lookup bidirectional LSTMs for details.
            bidirectional = False
            )
        
        middle_nodes = int(self.hidden_size / 2)

        self.fc1 = torch.nn.Linear(in_features = self.hidden_size, out_features = middle_nodes)
        self.fc2 = torch.nn.Linear(in_features = middle_nodes, out_features = 1)
        self.relu = torch.nn.functional.relu
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, batch):
        x = batch['input_ids'].to(device) # lookup token ids for our inputs
        x_lengths = batch['sequence_len'] # lookup lengths of our inputs
        embed_out = self.embedding(x) # get the embeddings of the token ids

        # In pytorch, RNN's need a packed sequence object as input
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            embed_out,
            x_lengths.tolist(),
            # use if sequences are sorted by length in the batch
            enforce_sorted = False, 
            batch_first = True
            )
        
        packed_out, (final_hidden_state, final_cell_state) = self.lstm(packed_input)

        # Inverse operation of pack_padded_sequence
        output,_ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first = True
            )
            
        lstm_out = output[range(len(output)), x_lengths - 1, :self.hidden_size]

        fc1_out = self.fc1(lstm_out)
        fc1_out = self.relu(fc1_out)
        fc2_out = self.fc2(fc1_out)
        final_out = self.sigmoid(fc2_out)
        return final_out
    
    def get_embedding_dims(self):
        return self.vocab_size, self.embedding_dim
    
"""
idk but the notebook had this in it
"""
config = {
    #model configurations
    'batch_size':32,
    'max_seq_length':100,
    'lr':1e-3,
    'label_count':2,
    'dropout_prob':2e-1,
    'hidden_size':256,
    'lstm_unit_cnt':2,

    #embeddings configurations
    'pretrained_embeddings':embs_npa,
    'freeze_embeddings':True,
    'vocab':vocab_npa,
    'pad_token':'<pad>',
    'unk_token':'<unk>',

    #data
    'train_df': df, #TODO: set val and test to appropriate
    'val_df': df,
    'test_df': df,
}



