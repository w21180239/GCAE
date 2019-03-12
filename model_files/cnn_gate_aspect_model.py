import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN_Gate_Aspect_Text(nn.Module):
    def __init__(self, args):
        super(CNN_Gate_Aspect_Text, self).__init__()
        self.args = args

        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        A = args.aspect_num

        Co = args.kernel_num
        Ks = args.kernel_sizes
        self.m = 10

        self.embed = nn.Embedding(V, D)
        self.embed.weight = nn.Parameter(args.embedding, requires_grad=True)

        self.aspect_embed = nn.Embedding(A, args.aspect_embed_dim)
        self.aspect_embed.weight = nn.Parameter(args.aspect_embedding, requires_grad=True)

        self.convs1 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.convs2 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])

        self.fc1 = nn.Linear(len(Ks) * Co, C)
        self.fc_aspect = nn.Linear(args.aspect_embed_dim, Co)

        self.matrix_size = args.matrix_size

        self.matrix = Variable(torch.Tensor(self.matrix_size, D))
        torch.nn.init.xavier_uniform(self.matrix, gain=1)

        self.mix_1 = nn.Linear(D+len(Ks) * Co,len(Ks) * Co)
        self.mix_2 = []
        for _ in range(self.m):
            self.mix_2.append(nn.Linear(int((D+len(Ks) * Co)/self.m),int((len(Ks) * Co)/self.m)).cuda())


    def forward(self, feature, aspect):
        inside_matrix = torch.stack([self.matrix[hash(str(feature[i])) % self.matrix_size] for i in range(len(feature))], 0).cuda()
        feature = self.embed(feature)  # (N, L, D)
        aspect_v = self.aspect_embed(aspect)  # (N, L', D)
        aspect_v = aspect_v.sum(1) / aspect_v.size(1)

        x = [F.tanh(conv(feature.transpose(1, 2))) for conv in self.convs1]  # [(N,Co,L), ...]*len(Ks)
        y = [F.relu(conv(feature.transpose(1, 2)) + self.fc_aspect(aspect_v).unsqueeze(2)) for conv in self.convs2]
        x = [i * j for i, j in zip(x, y)]

        # pooling method
        x0 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)
        x0 = [i.view(i.size(0), -1) for i in x0]

        x0 = torch.cat(x0, 1)
        h = self.mix_1(torch.cat([x0,inside_matrix],1))
        tmp=[]
        for i in range(self.m):
            l = self.mix_2[0].in_features
            tmp.append(self.mix_2[i](torch.cat([x0,inside_matrix],1)[:,i*l:(i+1)*l]))
        h = torch.cat(tmp,1)
        h = F.relu6(h)
        # x0 = x0 + inside_matrix
        x0 = h
        logit = self.fc1(x0)  # (N,C)
        return logit, x, y