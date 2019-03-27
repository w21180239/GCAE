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
        # Ks = [3,4,5]
        self.Ks = Ks
        self.m = 5

        self.embed = nn.Embedding(V, D)
        self.embed.weight = nn.Parameter(args.embedding, requires_grad=False)

        self.aspect_embed = nn.Embedding(A, args.aspect_embed_dim)
        self.aspect_embed.weight = nn.Parameter(args.aspect_embedding, requires_grad=False)

        self.convs2 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])


        self.convs_context = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.convs_asp = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.convs_con_asp = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.decons_context = nn.ModuleList([nn.ConvTranspose1d(Co,D,K) for K in Ks])
        self.decons_asp = nn.ModuleList([nn.ConvTranspose1d(Co, D, K) for K in Ks])
        self.decons_con_asp = nn.ModuleList([nn.ConvTranspose1d(Co, D, K) for K in Ks])


        self.fc1 = nn.Linear(len(Ks) * Co, C)
        self.fc_aspect = nn.Linear(args.aspect_embed_dim, Co)
        self.aspect_re = nn.Linear(len(Ks) * Co, len(Ks) * Co)

        self.matrix_size = args.matrix_size

        # self.matrix = Variable(torch.Tensor(self.matrix_size, D))
        # torch.nn.init.xavier_uniform(self.matrix, gain=1)
        #
        self.r = nn.Linear(2*len(Ks) * Co,len(Ks) * Co)
        self.z = nn.Linear(2* len(Ks) * Co, len(Ks) * Co)
        self._h = nn.Linear(2* len(Ks) * Co, len(Ks) * Co)

        self.linear_context  = nn.Linear(len(Ks) * Co, len(Ks) * Co)
        self.linear_con_asp  = nn.Linear(len(Ks) * Co, len(Ks) * Co)
        self.linear_asp = nn.Linear(len(Ks) * Co, len(Ks) * Co)

        self.reverse_context = nn.Linear(len(Ks) * Co, len(Ks) * Co)
        self.reverse_asp = nn.Linear(len(Ks) * Co, len(Ks) * Co)
        self.reverse_con_asp = nn.Linear(len(Ks) * Co, len(Ks) * Co)

        self.mix_2 = []
        for _ in range(self.m):
            self.mix_2.append(nn.Linear(int((D+len(Ks) * Co)/self.m),int((len(Ks) * Co)/self.m)).cuda())
        self.decoder_num = args.decoder_num
        hidden_width = [100*i for i in range(1,args.decoder_num+1)]
        self.decoder_list = nn.ModuleList([nn.Sequential(nn.Linear(len(Ks) * Co, h),nn.ReLU6(),nn.Linear(h,C)).cuda() for h in hidden_width])
        self.reconstruct = nn.Sequential(nn.Linear(len(Ks) * Co,500),nn.ReLU(),nn.Linear(500,D))

    def encode(self,input,convs,linear):
        x = [F.relu(conv(input.transpose(1, 2))) for conv in convs]  # [(N,Co,L), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = linear(x)
        return x

    def decode(self, input, deconvs, linear, length):
        x = linear(input)
        x = [x[:,i*100:(i+1)*100] for i in range(3)]
        x = [x[i].view(x[i].size(0),x[i].size(1),1).expand(x[i].size(0),x[i].size(1),length-self.Ks[i]+1) for i in range(len(x))]
        x = [deconvs[i](x[i]).transpose(1,2) for i in range(len(x))]
        x = sum(x)/len(x)
        return x


    def forward(self, feature, aspect):
        # o = feature
            # nn.Embedding
        feature = self.embed(feature)  # (N, L, D)
        aspect_v = self.aspect_embed(aspect)  # (N, L', D)
        aspect_v = aspect_v.sum(1) / aspect_v.size(1)
        copied_aspect_v = aspect_v.view(aspect_v.size(0),1,aspect_v.size(1)).expand_as(feature)
        length=feature.size(1)

        context = self.encode(feature, self.convs_context, self.linear_context)
        con_asp = self.encode(feature, self.convs_con_asp, self.linear_con_asp)
        asp = self.encode(copied_aspect_v, self.convs_asp, self.linear_asp)

        up_h = context+con_asp
        up_h = self.decode(up_h,self.decons_context,self.reverse_context,length)
        out_h = context+asp
        down_h = self.decode(out_h,self.decons_context,self.reverse_context,length)
        up_context = self.encode(up_h, self.convs_context, self.linear_context)
        up_con_asp = self.encode(up_h, self.convs_con_asp, self.linear_con_asp)
        down_asp = self.encode(down_h, self.convs_asp, self.linear_asp)
        reconstruct_aspect_v = up_con_asp
        down_final = down_asp+up_context
        reconstruct_feature = self.decode(down_final,self.decons_context,self.reverse_context,length)




        x = [F.relu(conv(feature.transpose(1, 2))) for conv in self.convs_context]
        y = [self.fc_aspect(aspect_v).unsqueeze(2) for conv in self.convs2]
        # x0 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)
        # x0 = [i.view(i.size(0), -1) for i in x0]
        # y0 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in y]  # [(N,Co), ...]*len(Ks)
        # y0 = [i.view(i.size(0), -1) for i in y0]
        # x0 = torch.cat(x0, 1)
        # y0 = torch.cat(y0, 1)
        # r = self.r(torch.cat([x0,y0],1))
        # r = F.sigmoid(r)
        # z = self.z(torch.cat([x0, y0], 1))
        # z = F.sigmoid(z)
        # _h = self._h(torch.cat([r*x0, y0], 1))
        # x0 = (1 - z) * x0 + z * _h
        #
        # # x = [F.tanh(conv(feature.transpose(1, 2))) for conv in self.convs1]  # [(N,Co,L), ...]*len(Ks)
        # # y = [F.relu(conv(feature.transpose(1, 2)) + self.fc_aspect(aspect_v).unsqueeze(2)) for conv in self.convs2]
        # # x = [i*j for i, j in zip(x, y)]
        # #
        # # # pooling method
        # # x0 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)
        # # x0 = [i.view(i.size(0), -1) for i in x0]
        # #
        # # x0 = torch.cat(x0, 1)
        # re_aspect = self.reconstruct(x0)
        # x0 = F.dropout(x0)
        out = out_h
        logit = self.fc1(out)  # (N,C)
        length = out.size(1)//self.decoder_num
        decoder_result = [list(self.decoder_list)[i](out) for i in range(len(list(self.decoder_list)))]
        # decoder_result = [decoder(x0[:,]) for decoder in list(self.decoder_list)]
        return logit, x, y,decoder_result,reconstruct_aspect_v,aspect_v,reconstruct_feature,feature