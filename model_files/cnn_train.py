import os
import sys
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
from attentionbagging import AttentionBagging

def bagging_train(train_iter,dev_iter,model,args):
    time_stamps = []
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.l2, lr_decay=args.lr_decay)

    steps = 0
    model.train()
    start_time = time.time()
    dev_acc, mixed_acc = 0, 0
    for i in range(model.bagging_num):
        for epoch in range(1, args.epochs+1):
            for _,batch in enumerate(train_iter):
                feature, target = batch
                if len(feature) < 2:
                    continue
                if args.cuda:
                    feature, target = feature.cuda(), target.cuda()

                bagging,logit = model(feature)

                out = bagging[i]
                loss = F.cross_entropy(out, target)
                loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()


                steps += 1
                if steps % args.log_interval == 0:
                    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                    accuracy = 100.0 * corrects/target.size(0)
                    if args.verbose == 1:
                        sys.stdout.write(
                            '\rEpoch: {}  Batch[{}] - loss: {:.6f}  acc: {:.4f}%'.format(epoch,steps,
                                                                                     loss.data,
                                                                                     accuracy,
                                                                                     ))


                if steps % args.save_interval == 0:
                    if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
                    save_prefix = os.path.join(args.save_dir, 'snapshot')
                    save_path = '{}_steps{}.pt'.format(save_prefix, steps)
                    torch.save(model, save_path)

            if epoch == args.epochs:
                dev_acc = bagging_eval(dev_iter, model,i, args)


            if args.verbose == 1:
                delta_time = time.time() - start_time
                time_stamps.append((dev_acc, delta_time))

def bagging_eval(data_iter, model,num, args):
    model.eval()

    final_corrects,final_avg_loss,now_corrects,now_avg_loss = 0,0,0,0

    for _,batch in enumerate(data_iter):
        feature, target = batch
        if len(feature) < 2:
            continue
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        bagging,final_out = model(feature)

        final_loss = F.cross_entropy(final_out, target, size_average=False)
        final_avg_loss += final_loss.data
        final_corrects += (torch.max(final_out, 1)
                     [1].view(target.size()).data == target.data).sum()

        now_loss = F.cross_entropy(bagging[num], target, size_average=False)
        now_avg_loss += now_loss.data
        now_corrects += (torch.max(bagging[num], 1)
                           [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    final_avg_loss = final_loss.data/size
    xx = int(final_corrects.cpu().numpy())
    final_accuracy = 100.0 * xx/size
    now_avg_loss = now_loss.data/size
    xx = int(now_corrects.cpu().numpy())
    now_accuracy = 100.0 * xx/size
    model.train()
    if args.verbose:
        print('\nEvaluation   bagging_loss: {:.6f}  bagging_acc: {:.4f}%({}/{}   now_loss: {:.6f}  now_acc: {:.4f}%({}/{})  num:{}'.format(
            final_avg_loss, final_accuracy, final_corrects, size,now_avg_loss, now_accuracy, now_corrects, size,num))
    return final_accuracy

def generate_bagging_iter(train_iter, dev_iter,model, args):
    train=[]
    train_tar = []
    dev=[]
    dev_tar = []
    for batch in train_iter:
        feature, aspect, target = batch.text, batch.aspect, batch.sentiment

        feature.data.t_()
        if len(feature) < 2:
            continue
        if not args.aspect_phrase:
            aspect.data.unsqueeze_(0)
        aspect.data.t_()
        target.data.sub_(1)  # batch first, index align
        train_tar.append(target)
        if args.cuda:
            feature, aspect, target, = feature.cuda(), aspect.cuda(), target.cuda()

        _, bagging_input = model(feature, aspect)
        train.append(bagging_input)
    for batch in dev_iter:
        feature, aspect, target = batch.text, batch.aspect, batch.sentiment
        feature.data.t_()
        if len(feature) < 2:
            continue
        if not args.aspect_phrase:
            aspect.data.unsqueeze_(0)
        aspect.data.t_()
        target.data.sub_(1)  # batch first, index align
        dev_tar.append(target)

        if args.cuda:
            feature, aspect, target, = feature.cuda(), aspect.cuda(), target.cuda()

        _, bagging_input = model(feature, aspect)
        dev.append(bagging_input)
    train = torch.cat(train,0)
    train_tar = torch.cat(train_tar,0)
    dev=dev[0]
    dev_tar = dev_tar[0]
    train_dataset = TensorDataset(train,train_tar)
    dev_dataset = TensorDataset(dev, dev_tar)
    train_loader = DataLoader(train_dataset,256,True)
    dev_loader = DataLoader(dev_dataset, 256, True)
    return train_loader,dev_loader



def train(train_iter, dev_iter, mixed_test_iter, model, args, text_field, aspect_field, sm_field, predict_iter):
    time_stamps = []
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.l2, lr_decay=args.lr_decay)

    steps = 0
    model.train()
    start_time = time.time()
    dev_acc, mixed_acc = 0, 0

    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            feature, aspect, target = batch.text, batch.aspect, batch.sentiment

            feature.data.t_()
            if len(feature) < 2:
                continue
            if not args.aspect_phrase:
                aspect.data.unsqueeze_(0)
            aspect.data.t_()
            target.data.sub_(1)  # batch first, index align

            if args.cuda:
                feature, aspect, target, = feature.cuda(), aspect.cuda(), target.cuda()

            optimizer.zero_grad()
            logit,_ = model(feature, aspect)

            loss = F.cross_entropy(logit, target)
            loss.backward(retain_graph=True)

            optimizer.step()
            optimizer.zero_grad()



            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                if args.verbose == 1:
                    sys.stdout.write(
                        '\rEpoch: {}  Batch[{}] - loss: {:.6f}  acc: {:.4f}%'.format(epoch,steps,
                                                                                 loss.data,
                                                                                 accuracy,
                                                                                 ))


            if steps % args.save_interval == 0:
                if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
                save_prefix = os.path.join(args.save_dir, 'snapshot')
                save_path = '{}_steps{}.pt'.format(save_prefix, steps)
                torch.save(model, save_path)

        if epoch == args.epochs:
            dev_acc = eval(dev_iter, model, args)
            # if mixed_test_iter:
            #     mixed_acc, _, _ = eval(mixed_test_iter, model, args)
            # else:
            #     mixed_acc = 0.0

            if args.verbose == 1:
                delta_time = time.time() - start_time
                # print('\n{:.4f} - {:.4f} - {:.4f}'.format(dev_acc, mixed_acc, delta_time))
                time_stamps.append((dev_acc, delta_time))
                # print()
    train_loader, dev_loader = generate_bagging_iter(train_iter,dev_iter,model,args)
    bagging_model = AttentionBagging(300,4,20,0.3,True)
    if args.cuda:
        bagging_model.cuda()
    bagging_train(train_loader,dev_loader,bagging_model,args)

    return (dev_acc, mixed_acc), time_stamps


def eval(data_iter, model, args):
    model.eval()
    global m1,m2
    # m2 = model.matrix.cpu().numpy()

    corrects, avg_loss,bagging_corrects,bagging_avg_loss = 0, 0,0,0
    loss = None

    for batch in data_iter:
        feature, aspect, target = batch.text, batch.aspect, batch.sentiment
        feature.data.t_()
        if not args.aspect_phrase:
            aspect.data.unsqueeze_(0)
        aspect.data.t_()
        target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, aspect, target = feature.cuda(), aspect.cuda(), target.cuda()

        logit,_ = model(feature, aspect)
        loss = F.cross_entropy(logit, target, size_average=False)
        avg_loss += loss.data
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()


    size = len(data_iter.dataset)
    avg_loss = loss.data/size
    xx= int(corrects.cpu().numpy())
    accuracy = 100.0 * xx/size
    model.train()
    if args.verbose:
        print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(
           avg_loss, accuracy, corrects, size,bagging_avg_loss))
    return accuracy