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
        self.m = 5

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

        self.r = nn.Linear(D+len(Ks) * Co,len(Ks) * Co)
        self.z = nn.Linear(D + len(Ks) * Co, len(Ks) * Co)
        self._h = nn.Linear(D + len(Ks) * Co, len(Ks) * Co)
        self.mix_2 = []
        for _ in range(self.m):
            self.mix_2.append(nn.Linear(int((D+len(Ks) * Co)/self.m),int((len(Ks) * Co)/self.m)).cuda())


    def forward(self, feature, aspect):
        o = feature
            # nn.Embedding
        index = [hash(str(o[i])) % self.matrix_size for i in range(len(o))]
        inside_matrix = torch.stack([self.matrix[hash(str(o[i])) % self.matrix_size] for i in range(len(o))], 0).cuda()
        feature = self.embed(feature)  # (N, L, D)
        aspect_v = self.aspect_embed(aspect)  # (N, L', D)
        aspect_v = aspect_v.sum(1) / aspect_v.size(1)

        x = [F.relu(conv(feature.transpose(1, 2))) for conv in self.convs1]  # [(N,Co,L), ...]*len(Ks)
        y = [F.relu(self.fc_aspect(aspect_v).unsqueeze(2)) for conv in self.convs2]
        # x = [i * j for i, j in zip(x, y)]

        # pooling method
        x0 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)
        x0 = [i.view(i.size(0), -1) for i in x0]
        y0 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in y]  # [(N,Co), ...]*len(Ks)
        y0 = [i.view(i.size(0), -1) for i in y0]
        x0 = torch.cat(x0, 1)
        y0 = torch.cat(y0, 1)
        r = self.r(torch.cat([x0,y0],1))
        r = F.tanh(r)
        z = self.z(torch.cat([x0, y0], 1))
        z = F.tanh(z)
        _h = self._h(torch.cat([r*x0, y0], 1))
        new_h = (1 - z) * x0 + z * _h
        # for i in range(_h.size(0)):
        #     self.matrix[index[i]] = new_h[i]

        # tmp=[]
        # for i in range(self.m):
        #     l = self.mix_2[0].in_features
        #     tmp.append(self.mix_2[i](torch.cat([x0,torch.stack([self.matrix[hash(str(o[i])) % self.matrix_size] for i in range(len(o))], 0).cuda()],1)[:,i*l:(i+1)*l]))
        # h = torch.cat(tmp,1)
        # h = F.relu6(h)
        # # x0 = x0 + inside_matrix
        # x0 = h

        logit = self.fc1(new_h)  # (N,C)
        return logit, x, y