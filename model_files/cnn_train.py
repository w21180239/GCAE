import os
import sys
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import numpy as np


def bagging_train(train_iter,dev_iter,model,args):
    time_stamps = []
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.l2, lr_decay=args.lr_decay)

    steps = 0
    model.train()
    start_time = time.time()
    dev_acc, mixed_acc = 0, 0
    for i in range(model.num):
        for epoch in range(1, args.epoch+1):
            for _,batch in enumerate(train_iter):
                feature, target = batch
                feature,target = torch.Tensor(feature),torch.Tensor(target)

                if args.cuda:
                    feature, target = feature.cuda(), target.cuda()

                logit,bagging = model(feature)

                out = bagging[i]
                loss = F.cross_entropy(out, target)
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

            if epoch == args.epoch:
                dev_acc = bagging_eval(dev_iter, model,i, args)


            if args.verbose == 1:
                delta_time = time.time() - start_time
                time_stamps.append((dev_acc, delta_time))

def bagging_eval(data_iter, model,num, args):
    model.eval()

    final_corrects,final_avg_loss,now_corrects,now_avg_loss = 0,0,0,0

    for _,batch in enumerate(data_iter):
        feature, target = batch
        feature, target = torch.Tensor(feature), torch.Tensor(target)

        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        final_out,bagging = model(feature)

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
        print('\nEvaluation   bagging_loss: {:.6f}  bagging_acc: {:.4f}%({}/{}   now_loss: {:.6f}  now_acc: {:.4f}%({}/{})'.format(
            final_avg_loss, final_accuracy, final_corrects, size,now_avg_loss, now_accuracy, now_corrects, size))
    return final_accuracy

def generate_bagging_iter(train_iter, dev_iter,model, args):
    train=[]
    train_tar = []
    dev=[]
    dev_tar = []
    for batch in train_iter:
        feature, aspect, target = batch.text, batch.aspect, batch.sentiment
        train_tar.append(target)
        feature.data.t_()
        if len(feature) < 2:
            continue
        if not args.aspect_phrase:
            aspect.data.unsqueeze_(0)
        aspect.data.t_()
        target.data.sub_(1)  # batch first, index align

        if args.cuda:
            feature, aspect, target, = feature.cuda(), aspect.cuda(), target.cuda()

        _, bagging_input = model(feature, aspect)
        train.append(bagging_input)
    for batch in dev_iter:
        feature, aspect, target = batch.text, batch.aspect, batch.sentiment
        dev_tar.append(target)
        feature.data.t_()
        if len(feature) < 2:
            continue
        if not args.aspect_phrase:
            aspect.data.unsqueeze_(0)
        aspect.data.t_()
        target.data.sub_(1)  # batch first, index align

        if args.cuda:
            feature, aspect, target, = feature.cuda(), aspect.cuda(), target.cuda()

        _, bagging_input = model(feature, aspect)
        dev.append(bagging_input)
    train_dataset = TensorDataset(train,train_tar)
    dev_dataset = TensorDataset(dev, dev_tar)
    train_loader = DataLoader(train_dataset,128,True)
    dev_loader = DataLoader(dev_dataset, 128, True)
    return train_loader,dev_loader



def train(train_iter, dev_iter, mixed_test_iter, model, args, text_field, aspect_field, sm_field, predict_iter):
    time_stamps = []
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.l2, lr_decay=args.lr_decay)

    steps = 0
    model.train()
    start_time = time.time()
    dev_acc, mixed_acc = 0, 0
    for name, p in model.named_parameters():
        if name.split('.')[0]=='attention_bagging':
            p.requires_grad = False
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
            logit, _, _, _,_ = model(feature, aspect)

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
            dev_acc, _, _ = eval(dev_iter, model, args)
            # if mixed_test_iter:
            #     mixed_acc, _, _ = eval(mixed_test_iter, model, args)
            # else:
            #     mixed_acc = 0.0

            if args.verbose == 1:
                delta_time = time.time() - start_time
                # print('\n{:.4f} - {:.4f} - {:.4f}'.format(dev_acc, mixed_acc, delta_time))
                time_stamps.append((dev_acc, delta_time))
                # print()
    for name, p in model.named_parameters():
        if name.split('.')[0] != 'attention_bagging':
            p.requires_grad = False
        else:
            p.requires_grad = True
    for ii in range(10):
        for epoch in range(1, 3):
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
                    feature, aspect, target = feature.cuda(), aspect.cuda(), target.cuda()

                optimizer.zero_grad()
                _, _, _, logit,bagging = model(feature, aspect)

                out = bagging[ii]
                loss = F.cross_entropy(out, target)
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

            if epoch == 2:
                dev_acc, _, _ = eval(dev_iter, model, args)


            if args.verbose == 1:
                delta_time = time.time() - start_time
                time_stamps.append((dev_acc, delta_time))
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

        logit, pooling_input, relu_weights,bagging_out,bagging = model(feature, aspect)
        loss = F.cross_entropy(logit, target, size_average=False)
        bagging_loss = F.cross_entropy(bagging_out, target, size_average=False)
        avg_loss += loss.data
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()
        bagging_avg_loss += bagging_loss.data
        bagging_corrects += (torch.max(bagging_out, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = loss.data/size
    bagging_avg_loss = bagging_loss.data/size
    xx= int(corrects.cpu().numpy())
    accuracy = 100.0 * xx/size
    xx = int(bagging_corrects.cpu().numpy())
    bagging_accuracy = 100.0 * xx/size
    model.train()
    if args.verbose:
        print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{})  bagging_loss: {:.6f}  bagging_acc: {:.4f}%({}/{})'.format(
           avg_loss, accuracy, corrects, size,bagging_avg_loss, bagging_accuracy, bagging_corrects, size))
    return accuracy, pooling_input, relu_weights