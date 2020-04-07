import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from codes.stats_scripts import on_gpu, plotting, print_stats
from sklearn.cluster import KMeans, AgglomerativeClustering



def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch%1==0:
        if lr>0.00001:
            lr = lr-0.000005
    # if epoch==10:
    #     lr = 0.0001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


gpu = on_gpu()
criterion1 = nn.MSELoss(reduction='sum')  # Reconstruction loss


def pretrain_lstm(epoch_nb, encoder, decoder, loader, args):
    learning_rate = args.learning_rate
    optimizer = torch.optim.Adam((list(encoder.parameters()) + list(decoder.parameters())), lr=learning_rate)
    # optimizer = torch.optim.SGD((list(encoder.parameters()) + list(decoder.parameters())), lr=args.learning_rate, momentum=0.5)
    # print_stats(args.stats_file, "Optimiser SGD")
    encoder.train()
    decoder.train()
    epoch_loss_list = []
    for epoch in range(epoch_nb):
        learning_rate = adjust_learning_rate(optimizer, epoch+1, learning_rate)
        print("learning rate = " + str(learning_rate))

        total_loss = 0
        for batch_idx, (data, _, id) in enumerate(loader):
            if gpu:
                data = data.cuda()
            # we delete zero-padding so the whole batch have the sequence lenght equal to the longest sequence in this batch
            # initially the sequence lenght is equal to the lenght of SITS
            idx = [i for i in range(data.size(1) - 1, -1, -1)]
            idx = torch.LongTensor(idx).cuda()
            inverted_data = torch.index_select(data, 1, idx)
            # inverted_data = np.flip(data, axis=1)
            encoded_output = encoder(Variable(data))
            decoded = decoder(encoded_output)
            loss = criterion1(decoded, Variable(inverted_data))
            loss_data = loss.item()
            total_loss += loss_data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (batch_idx+1) % 200 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.7f}'.format(
                    (epoch+1), (batch_idx+1) * args.batch_size, len(loader)*args.batch_size,
                    100. * (batch_idx+1) / len(loader), loss_data))
        epoch_loss = total_loss / len(loader)
        epoch_loss_list.append(epoch_loss)
        epoch_stats = "Pretraining Epoch {} Complete: Avg. Loss: {:.7f}".format(epoch + 1, epoch_loss)
        print_stats(args.stats_file, epoch_stats)
        if (epoch) % 5 == 0:
            plotting(epoch+1, np.asarray(epoch_loss_list), args.path_results)
            torch.save([encoder, decoder], (args.path_model + 'ae-model_ep_' + str(epoch + 1) + "_loss_" + str(
                round(epoch_loss, 7)) + args.run_name + '.pkl'))
    try:
        plotting(epoch + 1, np.asarray(epoch_loss_list), args.path_results)
    except UnboundLocalError:
        pass
    try:
        torch.save([encoder, decoder], (args.path_model + 'ae-model_ep_' + str(epoch + 1) + "_loss_" + str(round(epoch_loss, 7)) + args.run_name + '.pkl'))
    except:
        pass



# K-means clusters initialisation
def encode_lstm(encoder, W, loader_enc, args, cl_nb=None):
    encoder = encoder.eval()
    hidden_array = None
    batch_size = W
    for batch_idx, (data, _, id) in enumerate(loader_enc):
        # data = pad_sequence(data, batch_first=True)
        if gpu:
            data = data.cuda()
        # we delete zero-padding so the whole batch have the sequence lenght equal to the longest sequence in this batch
        # initially the sequence lenght is equal to the lenght of SITS
        el = 0
        condition = True
        while condition:
            if torch.sum(data[:, el - 1]) == 0:
                el -= 1
            else:
                condition = False
        if el != 0:
            data = data[:, :data.size(1) + el]
        _, hidden, _ = encoder(Variable(data))
        if (batch_idx + 1) % 2 == 0:
            print('Initializing K-means: {}/{} ({:.0f}%)'.format(
                (batch_idx + 1) * batch_size, len(loader_enc) * batch_size,
                             100. * (batch_idx + 1) / len(loader_enc)))
        if hidden_array is not None:
            hidden_array = np.concatenate((hidden_array, hidden.cpu().detach().numpy()), 0)
        else:
            hidden_array = hidden.cpu().detach().numpy()
    # if we perform clustering for a concrete number of clusters
    if cl_nb is None:
        kmeans = KMeans(n_clusters=args.nb_clusters)
        cluster_h = AgglomerativeClustering(n_clusters=args.nb_clusters, affinity='euclidean', linkage='ward')
        labels = kmeans.fit_predict(hidden_array)
        labels_h = cluster_h.fit_predict(hidden_array)
        return labels, labels_h, hidden_array
    # if we perform clustering for a range of number of clusters (as in the article)
    else:
        labels_all = []
        labels_h_all = []
        for cl in cl_nb:
            kmeans = KMeans(n_clusters=cl)
            cluster_h = AgglomerativeClustering(n_clusters=cl, affinity='euclidean', linkage='ward')
            labels = kmeans.fit_predict(hidden_array)
            labels_h = cluster_h.fit_predict(hidden_array)
            labels_all.append(labels)
            labels_h_all.append(labels_h)
        return labels_all, labels_h_all, hidden_array






# K-means clusters initialisation
def encode_lstm(encoder, W, loader_enc, args, cl_nb=None):
    # initialize_clusters
    encoder = encoder.eval()
    hidden_array = None
    batch_size = W
    for batch_idx, (data, _, id) in enumerate(loader_enc):
        if gpu:
            data = data.cuda()
        el = 0
        condition = True
        while condition:
            if torch.sum(data[:, el - 1]) == 0:
                el -= 1
            else:
                condition = False
        if el != 0:
            data = data[:, :data.size(1) + el]
        encoded, hidden, _ = encoder(Variable(data))
        if (batch_idx + 1) % 2 == 0:
            print('Initializing K-means: {}/{} ({:.0f}%)'.format(
                (batch_idx + 1) * batch_size, len(loader_enc) * batch_size,
                             100. * (batch_idx + 1) / len(loader_enc)))
        if hidden_array is not None:
            hidden_array = np.concatenate((hidden_array, hidden.cpu().detach().numpy()), 0)
        else:
            hidden_array = hidden.cpu().detach().numpy()

    if cl_nb is None:
        kmeans = KMeans(n_clusters=args.nb_clusters)
        cluster_h = AgglomerativeClustering(n_clusters=args.nb_clusters, affinity='euclidean', linkage='single')
        labels = kmeans.fit_predict(hidden_array)
        labels_h = cluster_h.fit_predict(hidden_array)
        return labels, labels_h, hidden_array
    else:
        labels_all = []
        labels_h_all = []
        for cl in cl_nb:
            kmeans = KMeans(n_clusters=cl)
            cluster_h = AgglomerativeClustering(n_clusters=cl, affinity='euclidean', linkage='ward')
            labels = kmeans.fit_predict(hidden_array)
            labels_h = cluster_h.fit_predict(hidden_array)
            labels_all.append(labels)
            labels_h_all.append(labels_h)
        return labels_all, labels_h_all, hidden_array

#
#
# # Encoder function for the pretrained model
# def encode_lstm(encoder, W, loader_enc):
#     # initialize_clusters
#     encoder = encoder.eval()
#     hidden_array = None
#     batch_size = W
#
#     for batch_idx, (data, _, id) in enumerate(loader_enc):
#         if gpu:
#             data = data.cuda()
#         encoded, hidden, _ = encoder(Variable(data))
#         if (batch_idx + 1) % 2 == 0:
#             print('Encoding: {}/{} ({:.0f}%)'.format(
#                 (batch_idx + 1) * batch_size, len(loader_enc) * batch_size,
#                              100. * (batch_idx + 1) / len(loader_enc)))
#         if hidden_array is not None:
#             hidden_array = np.concatenate((hidden_array, hidden.cpu().detach().numpy()), 0)
#         else:
#             hidden_array = hidden.cpu().detach().numpy()
#
#
#     # if cl_nb is None:
#     #     kmeans = KMeans(n_clusters=args.nb_clusters)
#     #     cluster_h = AgglomerativeClustering(n_clusters=args.nb_clusters, affinity='euclidean', linkage='ward')
#     #     labels = kmeans.fit_predict(hidden_array)
#     #     labels_h = cluster_h.fit_predict(hidden_array)
#     #     #labels = kmeans.predict(hidden_array)
#     #     visualize_tsne(hidden_array, labels,  "initializing centers", args.path_results)
#     #     centers = kmeans.cluster_centers_
#     #     weights = torch.from_numpy(centers)
#     #     if gpu:
#     #         weights = weights.cuda()
#     #     clustering.set_weight(weights)
#     #     return labels, labels_h, hidden_array
#     # else:
#     #     labels_all = []
#     #     labels_h_all = []
#     #     for cl in cl_nb:
#     #         kmeans = MiniBatchKMeans(n_clusters=cl, batch_size=5000, verbose=0, max_no_improvement=10)
#     #         kmeans = kmeans.fit(hidden_array)
#     #         labels = kmeans.predict(hidden_array)
#     #
#     #         # memory = Memory("/media/user/DATA/Results/TS_clustering/memory_cache/", verbose=0)
#     #         # cluster_h = AgglomerativeClustering(n_clusters=cl, memory=memory)
#     #         # labels_h = cluster_h.fit_predict(hidden_array)
#     #         # memory.clear(warn=False)
#     #
#     #         # Z = linkage(hidden_array, 'ward')
#     #         # labels_h = fcluster(Z, t=cl, criterion='maxclust')
#     #
#     #
#     #         # visualize_tsne(hidden_array, labels, "initializing centers", args.path_results)
#     #         # labels_all.append(labels)
#     #         labels_h_all.append(labels)
#     #     # return labels_all, labels_h_all, hidden_array
#     #     # return None, labels_h_all, hidden_array
#     return None, None, hidden_array

