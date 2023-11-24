import torch
import torch.nn as nn
import torch.nn.functional as F

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
def DensewithBN(in_fea, out_fea, normalize=True):
    layers=[nn.Linear(in_fea, out_fea)]
    if normalize==True:
        layers.append(nn.BatchNorm1d(num_features = out_fea))
    layers.append(nn.ReLU())
    return layers

class multi_triples_lstm(nn.Module):
    def __init__(self,
                 vocab,
                 hidden_dim: int=32,
                 embedding_dim: int=5,
                 object_embedding_dim: int=5,
                 predicate_embedding_dim: int=5,
                 ):
        super(multi_triples_lstm,self).__init__()
        self.hidden_dim = hidden_dim

        object_categories = len(vocab["object_name_to_idx"])
        predicate_categories = len(vocab["pred_name_to_idx"])

        self.object_embedding = nn.Embedding(num_embeddings=object_categories, embedding_dim=object_embedding_dim) # (from 25->5)
        self.predicate_embedding = nn.Embedding(num_embeddings=predicate_categories, embedding_dim=predicate_embedding_dim) # (from 25->5)

        self.output_fc1 = nn.Sequential(*DensewithBN(object_embedding_dim+predicate_embedding_dim+2+4, hidden_dim))
        self.output_fc2 = nn.Linear(hidden_dim, 4)

        self.lstm = nn.LSTM(hidden_dim, hidden_dim)# input dim = 32, output dim = 32
        self.lstm_cell = nn.LSTMCell(object_embedding_dim+predicate_embedding_dim+2+4, hidden_dim)

    def concatenate_features(self, input_object, input_box, predicate, subject_indicator):
        input_object_emb = self.object_embedding(input_object) # (None,e1)
        predicate_emb = self.predicate_embedding(predicate) # (None,e2)
        subject_indicator = F.one_hot(subject_indicator) # (None,2)

        output = torch.cat([input_object_emb, predicate_emb, subject_indicator, input_box], dim=1) # (None,5+2+2)
        return output

    def lstmforward(self, inputs, batch_size, return_sequence=False): # inputs: (None,max_len,hidden_dim), binary: (None,1)
        """ implement lstm"""
        hiddens = []
        c_memories = []
        h = FloatTensor(batch_size,self.hidden_dim).zero_()
        c = FloatTensor(batch_size,self.hidden_dim).zero_()
        #textcode: (batch_size,seq_len) inputs: (batch_size, seq_len, embedding_dim)

        for idx in range(inputs.shape[1]):
            h, c = self.lstm_cell(inputs[:,idx,:], (h,c))
            hiddens.append(h.unsqueeze(1))
            c_memories.append(c.unsqueeze(1))
        if return_sequence==True:
            return torch.cat(hiddens, dim=1)
        else:
            return hiddens[-1]

    def forward(self, input_objects, input_boxes, predicates, subject_indicators, target_object):
        batch_size = predicates.shape[0]
        fc_hiddens = []
        for idx in range(predicates.shape[-1]):
            embeddings = self.concatenate_features(input_objects[:,idx], input_boxes[:,idx],
                                                   predicates[:,idx], subject_indicators[:,idx])
            fc_hiddens.append(embeddings.unsqueeze(dim=1))

        inputs = torch.cat(fc_hiddens, dim=1) #(None, max_triples_num, object_emb+predicate_emb+box+subject_indicator), (None,7,16)
        output = self.lstmforward(inputs,batch_size) # (None,1,embed_dim)
        output = self.output_fc2(output[:,0,:])
        output = torch.clamp(output,max=1, min=0)
        return output



