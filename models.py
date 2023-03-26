import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

# You can use this class to define your model
class CNNModel(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, mode):
        print(f'CNNModel vocab_size: {vocab_size}, embedding_dim: {embedding_dim}, n_filters: {n_filters}, filter_sizes: {filter_sizes}, mode: {mode}')
        super(CNNModel, self).__init__()
        self.mode = mode
        self.embedding = nn.Sequential()
        self.embedding.add_module('f_embed', nn.Embedding(vocab_size, embedding_dim))
     
        self.feature = nn.Sequential()
        self.feature.add_module('f_convs', nn.ModuleList(
            [
                nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim)) 
                for fs in filter_sizes
            ]
        ))
        self.feature.add_module('f_drop', nn.Dropout2d())
        
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(300, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 2))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax())
        
        if self.mode != 'TLnoDA':
            self.domain_classifier = nn.Sequential()
            self.domain_classifier.add_module('d_fc1', nn.Linear(300, 100))
            self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
            self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
            self.domain_classifier.add_module('d_drop1', nn.Dropout())
            self.domain_classifier.add_module('d_fc2', nn.Linear(100, 100))
            self.domain_classifier.add_module('d_bn2', nn.BatchNorm1d(100))
            self.domain_classifier.add_module('d_relu2', nn.ReLU(True))
            self.domain_classifier.add_module('d_fc3', nn.Linear(100, 2))
            self.domain_classifier.add_module('d_softmax', nn.LogSoftmax())
        
    def init_weights(self, pretrained_word_vectors, is_static=False):
        self.embedding.f_embed.weight = nn.Parameter(pretrained_word_vectors)
        if is_static:
             self.embedding.f_embed.weight.requires_grad = False
        
    def forward(self, input_data, alpha):
        x = input_data.permute(1, 0)
        embedded = self.embedding.f_embed(x)
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.feature.f_convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.feature.f_drop(torch.cat(pooled, dim=1))
        feature = cat.view(-1, 300) # the size -1 is inferred from other dimensions, n_filters * len(filter_sizes)
        class_output = self.class_classifier(feature)
        if self.mode != 'TLnoDA':
            reverse_feature = ReverseLayerF.apply(feature, alpha)
            domain_output = self.domain_classifier(reverse_feature)   
            return class_output, domain_output
        else:
            return class_output, None