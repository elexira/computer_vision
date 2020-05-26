import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

def one_hot_encode(arr, n_labels):
    
    # Initialize the the encoded array
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    
    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    
    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    
    return one_hot


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

    def forward(self, images):
        features = self.resnet(images)
        print("output from resnet shape is: ", features.shape)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        input_embeddings = self.word_embeddings(captions)
#         print("input_embeddings: " , input_embeddings.shape)
#         print("input_embeddings: " , input_embeddings.shape)
#         print("lenght of caption: ", len(caption)) 
        initia_hidden = torch.cat((features, features), 1).unsqueeze_(0)
#         print(initia_hidden.shape)
        lstm_output, (h , c)  = self.lstm(input_embeddings, (initia_hidden,initia_hidden) )
        fc_output = self.fc(lstm_output)
        output_probabilities = F.log_softmax(fc_output, dim=1)
        return output_probabilities

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass