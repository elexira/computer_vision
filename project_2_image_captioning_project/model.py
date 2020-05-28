import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F



class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        # Amir: i think the linear layer parameters needs training, the rest comes from Resnet pretrained. 
        # Amir: i think [:-1] means the final FC layer of resnet is not being considered and is removed 
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # adding a FC layer of of our own to the Resent (replacing Resnet final FC layter with this)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        features = self.resnet(images)
        # print("output from resnet shape is: ", features.shape)
        # print("feature shape 1 :", features.shape)
        features = features.view(features.size(0), -1)
        # print("feature shape 2 :", features.shape)
        features = self.embed(features)
        # print("feature shape 3 :", features.shape)
        features = self.bn(features)
        # print("feature shape 4 :", features.shape)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, dropout=0, num_layers=2):
        super(DecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size,
                            hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
        
    def forward(self, features, captions):
        # we do not input the <end> to the LSTM. the last word in caption is the last input which should predict <end>
        captions = captions[:, :-1]
        caption_embeddings = self.word_embeddings(captions)
        features = features.unsqueeze(1)
        lstm_input = torch.cat((features,  caption_embeddings), 1)
        lstm_output, (h, c) = self.lstm(lstm_input)
        dropout_output = self.dropout(lstm_output)
        fc_output = self.fc(dropout_output)
        output_probabilities = nn.Softmax(dim=2)(fc_output)
        return fc_output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        prediction_list = list()
        start_embedding = self.word_embeddings(torch.tensor([0]).to('cuda')).unsqueeze(0)
        # print("start_embedding:", start_embedding.shape)
        features_updated = inputs
        # print("features_updated:", features_updated.shape)
        # print("features_updated values:", features_updated[0, 0, 1:10])
        lstm_input = torch.cat((features_updated, start_embedding), 1)
        # print("lstm_input:", lstm_input.shape)
        # print("lstm_input values:", lstm_input[0, :, 0:10])
        lstm_output, (h, c) = self.lstm(lstm_input)
        # print("lstm_output, (h, c)", lstm_output.shape, h.shape, c.shape)
        # print("lstm_h values", h[0, 0, 1:10])
        # h.view(num_layers, num_directions, batch, hidden_size) # if you want to parse h
        fc_input = lstm_output[:, -1, :].reshape(1, self.hidden_size)
        # print("the input sending to fc", fc_input.shape)
        fc_output = self.fc(fc_input)  # i get 2 predictions, only want the last prediction, the first one is <start>
        # print("fc_output", fc_output.shape)
        prediction_index = torch.argmax(fc_output, dim=1)
        # print("prediction_index", prediction_index.shape)
        prediction_list.append(int(prediction_index.cpu().numpy()[0]))
        word_embedding = self.word_embeddings(prediction_index).unsqueeze(0)
        # print("word_embedding", word_embedding.shape)
        for pred in range(max_len):
            if prediction_index != 1:
                # check here if the prediction is not end
                lstm_output, (h, c) = self.lstm(word_embedding, (h, c))
                fc_output = self.fc(lstm_output.squeeze(0))
                prediction_index = torch.argmax(fc_output, dim=1)
                prediction_list.append(int(prediction_index.cpu().numpy()[0]))
                word_embedding = self.word_embeddings(prediction_index).unsqueeze(0)
        return prediction_list
    
    def init_weights(self):
        ''' Initialize weights for fully connected layer and lstm forget gate bias'''
        initrange = 0.1
        
        # Set bias tensor to all zeros
        self.fc.bias.data.fill_(0)
        # FC weights as random uniform
        self.fc.weight.data.uniform_(-1, 1)

        