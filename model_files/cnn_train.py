import os
import sys
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

m1 = np.ndarray((1000,300))
m2 = np.ndarray((1000,300))

def train(train_iter, dev_iter, mixed_test_iter, model, args, text_field, aspect_field, sm_field, predict_iter):
    global m1
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