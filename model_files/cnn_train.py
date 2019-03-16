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
    # m1 = model.matrix.cpu().numpy()
    optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.l2, lr_decay=args.lr_decay)

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
                feature, aspect, target = feature.cuda(), aspect.cuda(), target.cuda()

            optimizer.zero_grad()
            logit, _, _,decode_list,re,ori = model(feature, aspect)

            loss = F.cross_entropy(logit, target)
            ss = F.mse_loss(re, ori)
            loss = loss+args.support*ss
            loss.backward(retain_graph=True)

            optimizer.step()
            optimizer.zero_grad()

            # ss = F.mse_loss(re, ori)
            # ss.backward(retain_graph=True)
            #
            # optimizer.step()
            # optimizer.zero_grad()


            # for output in decode_list:
            #     ll = F.cross_entropy(output, target)
            #     ll.backward(retain_graph=True)
            #     optimizer.step()
            #     optimizer.zero_grad()

            # for name, p in model.named_parameters():
            #     p.requires_grad = True
            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                if args.verbose == 1:
                    sys.stdout.write(
                        '\rEpoch: {}  Batch[{}] - loss: {:.6f}  acc: {:.4f}%'.format(epoch,steps,
                                                                                 loss.data[0],
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
        if name.split('.')[0] != 'decoder_list':
            p.requires_grad = False
    optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.l2, lr_decay=args.lr_decay)
    for i in range(args.decoder_num):
        print("\nDecoder_%d"%(i))
        for epoch in range(1, args.decoder_epoch+1):
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
                logit, _, _,decode_list,re,ori = model(feature, aspect)

                ll = F.cross_entropy(decode_list[i], target)
                ll.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()


                steps += 1
                if steps % args.log_interval == 0:
                    corrects = (torch.max(decode_list[i], 1)[1].view(target.size()).data == target.data).sum()
                    accuracy = 100.0 * corrects/batch.batch_size
                    if args.verbose == 1:
                        sys.stdout.write(
                            '\rEpoch: {}  Batch[{}] - loss: {:.6f}  acc: {:.4f}%'.format(epoch,steps,
                                                                                     ll.data[0],
                                                                                     accuracy,
                                                                                     ))


                if steps % args.save_interval == 0:
                    if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
                    save_prefix = os.path.join(args.save_dir, 'snapshot')
                    save_path = '{}_steps{}.pt'.format(save_prefix, steps)
                    torch.save(model, save_path)

            if epoch == args.decoder_epoch:
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
    return (dev_acc, mixed_acc), time_stamps


def eval(data_iter, model, args):
    model.eval()
    global m1,m2
    # m2 = model.matrix.cpu().numpy()
    for i in range(1000):
        for j in range(300):
            if m1[i][j]!=m2[i][j]:
                xx=1
            else:
                xx=2
    corrects, avg_loss = 0, 0
    correct_list = [0 for a in range(args.decoder_num)]
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

        logit, pooling_input, relu_weights,decode_list,re,ori = model(feature, aspect)
        loss = F.cross_entropy(logit, target, size_average=False)
        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()
        for i in range(args.decoder_num):
            correct_list[i]+=(torch.max(decode_list[i], 1)
                     [1].view(target.size()).data == target.data).sum()
        softmax = torch.nn.Softmax(1)
        for i in range(args.decoder_num):
            decode_list[i] = softmax(decode_list[i])
        sum_correct = (torch.max(sum(decode_list), 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = loss.data[0]/size
    xx= int(corrects.cpu().numpy())
    accuracy = 100.0 * xx/size
    accuracy_list = [100.0 * int(hh.cpu().numpy())/size for hh in correct_list]
    total_accuracy = 100.0*int(sum_correct.cpu().numpy())/size
    model.train()
    if args.verbose:
        print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(
           avg_loss, accuracy, corrects, size))
        for i in range(len(accuracy_list)):
            print("acc_%d    %.4f    "%(i,accuracy_list[i]),end='')
        print("\ntotal_acc:%.4f"%(total_accuracy))
    return accuracy, pooling_input, relu_weights