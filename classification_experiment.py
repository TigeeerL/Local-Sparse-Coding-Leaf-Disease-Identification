import time
from pathlib import Path

import numpy
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.sparse as ss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sacred import Experiment
from sklearn.cluster import k_means
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.manifold import spectral_embedding
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm as progress_bar
from statistics import mean, median
import tensorflow as tf

from keras import layers
import keras

import torchvision

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from itertools import combinations

import model, utils
import datasets

import cvxpy as cp

###
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
###


ex = Experiment("Application of K-Deep-Simplex to YaleB Classification")


@ex.config
def cfg():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    hyp = {
        "num_layers": 100,
        "input_size": 256*256, # For YaleB: original 192 * 168. other possible choices: 40 * 35. MNIST is 28*28. For leaf: original 256 * 256
        "hidden_size": 400, # divisible by num_of_people to avoid bugs
        "penalty": 0.05,
        "train_step": True,
    }
    lr = 1e-3 # step size
    dataset = "leaf_disease_apple" # other options include: # "yale2" # "yale3" # "yale10" # "yale_full" # "yale_var" # "mnist5" # "mnist10" # 'leaf_disease_apple'
    num_of_people = 4 # this is for selecting the number of people used from the yale dataset
    quantity = 10000 # draw quantity amount of sample from the dataset
    train_amount = 0.5
    path_to_data = "./leaf_disease" # None
    subset = None
    epochs = 500 # number of passes. total iterations. one epoch goes through all data
    batch_size = 10000 # breakdown into smaller pieces. learn everything 10000
    workers = 4 # GPU computing
    # seed = 0
    feature_dim = 20
    image_size = [256,256]
    # dic_learning = False
    compare_poss = ['CVX_solver','dic','exact_dic', 'AG_aug', 'kmeans', 'original','reg']
    compare = np.array([[0,0,0,0,1,1,0]]) # if no augmentation, then original or not doesn't matter. For the sake of consistency, just use original. Does not support repeated elements
    expand_fac = 1 # fix this to 1 for now
    aug_rep = 1
    exp_rep = 5
    # init = "kmeans" # potential choices: random_subset, kmeans
    # init_data = "original" # potential choices: all, original
    baseline = "no" # potential choices: no, KM, SVM


@ex.automain
def run(
    _run,
    device,
    hyp,
    lr,
    dataset,
    quantity,
    train_amount,
    path_to_data,
    subset,
    epochs,
    batch_size,
    workers,
    seed,
    feature_dim,
    num_of_people,
    image_size,
    compare_poss,
    compare,
    # dic_learning,
    expand_fac,
    aug_rep,
    exp_rep,
    # init,
    # init_data,
    baseline,
):
    if path_to_data is None:
        raise Exception("must supply a path_to_data (e.g. python classification_experiment.py with path_to_data=../CroppedYale)")

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    combined_accuracy = []
    combined_accuracy_crit = []
    combined_accuracy_avg = []
    CVX_solve_flag = True
    for exp in range(exp_rep):
        combined_accuracy.append(np.array([]))
        combined_accuracy_crit.append(np.array([]))
        combined_accuracy_avg.append(np.array([]))

        # this is for comparing the x learned from different methods
        x_train_comp = [0] * compare.shape[0]
        x_test_comp = [0] * compare.shape[0]

        consistent_dic = torch.empty([])

        # data, labels, k = datasets.get_data(dataset, path_to_data, num_of_people, input_size = image_size, quantity = quantity) # whatever, get data. Can look at it later ### temporary

        ### for using CNN to extract features. This step should in fact be done after the data augmentation step
        img_width, img_height = 224, 224  # Default input size for VGG16

        # Instantiate convolutional base

        conv_base = VGG16(weights='imagenet',
                          input_shape=(img_width, img_height, 3))

        conv_base = Model(inputs=conv_base.inputs, outputs=conv_base.layers[-2].output)

        conv_base.summary()

        from keras.preprocessing.image import ImageDataGenerator

        # datagen = ImageDataGenerator(rescale=1. / 255)
        datagen = ImageDataGenerator()
        batch_size = 10000

        def extract_features(directory, sample_count):
            features = np.zeros(shape=(sample_count, 4096))  # Must be equal to the output of the convolutional base
            labels = np.zeros(shape=(sample_count, num_of_people))
            # Preprocess data
            generator = datagen.flow_from_directory(path_to_data,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')
            features = np.zeros(
                shape=(generator.samples, 4096))  # Must be equal to the output of the convolutional base
            labels = np.zeros(shape=(generator.samples, 4))
            # Pass data through convolutional base
            i = 0
            for inputs_batch, labels_batch in generator:
                inputs_batch = preprocess_input(inputs_batch)
                features_batch = conv_base.predict(inputs_batch)
                features[i * batch_size: (i + 1) * batch_size] = features_batch
                labels[i * batch_size: (i + 1) * batch_size] = labels_batch
                i += 1
                if i * batch_size >= sample_count:
                    break
            return features, labels

        data, labels = extract_features(path_to_data, quantity)  # Agree with our small dataset size # 3171 for leaf disease
        data = torch.tensor(data, dtype=torch.float32)
        labels = np.argmax(labels, axis=1)
        labels = torch.tensor(labels, dtype=torch.int32)
        k = num_of_people
        ###

        # train_data, train_labels, test_data, test_labels = utils.split_in_half(data, labels, k) # split into training data and testing data. Not going to worry about it now.
        train_data, train_labels, test_data, test_labels = utils.split_data(data, labels, k, train_amount)

        original_train_data = train_data
        original_train_labels = train_labels
        for config_num in range(compare.shape[0]):
            config = compare[config_num]
            print(config)

            # initialize configuration
            init = ""
            init_data = ""
            dic_learning = config[compare_poss.index('dic')]
            CVX_solver = config[compare_poss.index('CVX_solver')]
            exact_dic = config[compare_poss.index('exact_dic')]
            reg = ""
            if config[compare_poss.index('AG_aug')] == False:
                expand_fac = 0
            elif config[compare_poss.index('AG_aug')] == True:
                expand_fac = 1
            if config[compare_poss.index('kmeans')] == True:
                init = "kmeans"
            elif config[compare_poss.index('kmeans')] == False:
                init = "random_subset"
            if config[compare_poss.index('original')] == True:
                init_data = "original"
            elif config[compare_poss.index('original')] == False:
                init_data = "all"
            if config[compare_poss.index('reg')] == 0:
                reg = "ex1"
            elif config[compare_poss.index('reg')] == 1:
                reg = "abs"
            elif config[compare_poss.index('reg')] == 2:
                reg = "ex2"

            # data augmentation
            rotate = keras.Sequential([
                # tf.keras.layers.RandomBrightness(factor=0.2),
                # layers.RandomContrast(0.2),
                layers.RandomRotation(1/36),
                # layers.RandomFlip('horizontal')
            ])
            scale = keras.Sequential([
                # tf.keras.layers.RandomBrightness(factor=0.2),
                # layers.RandomContrast(0.2),
                layers.RandomZoom(0.1),
                # layers.RandomFlip('horizontal')
            ])

            aug_rep_accuracy = []
            quality = np.zeros(aug_rep)
            for i_out in range(aug_rep):
                all_train_data = []
                all_train_data.append(original_train_data)
                train_labels = original_train_labels
                train_data = original_train_data
                hflip = 1
                rot = 5
                scal = 5
                for l in range(expand_fac):
                    # fig, axs = plt.subplots(2, 6, subplot_kw={'xticks': [], 'yticks': []})
                    for i in range(train_data.shape[0]):
                        curr_image = torch.reshape(train_data[i, :], image_size).detach()
                        for op in range(hflip+rot+scal):
                            augmented_image = curr_image
                            if hflip == 1 and op == 0:
                                augmented_image = torchvision.transforms.functional.hflip(curr_image)
                            elif op < hflip + rot:
                                image = tf.convert_to_tensor(curr_image)
                                image = tf.expand_dims(image, -1)
                                augmented_image = rotate(image)
                                augmented_image = tf.squeeze(augmented_image)
                                augmented_image = torch.tensor(augmented_image.numpy())
                            elif op < hflip + rot + scal:
                                image = tf.convert_to_tensor(curr_image)
                                image = tf.expand_dims(image, -1)
                                augmented_image = scale(image)
                                augmented_image = tf.squeeze(augmented_image)
                                augmented_image = torch.tensor(augmented_image.numpy())
                        # image = tf.convert_to_tensor(image)
                        # image = tf.expand_dims(image, -1)
                        # augmented_image = data_augmentation(image)
                        # augmented_image = tf.squeeze(augmented_image)
                        # augmented_image = torch.tensor(augmented_image.numpy())
                            augmented_image = torch.reshape(augmented_image, (1, np.prod(image_size)))
                            # print(torch.linalg.norm(train_data[i,:]-augmented_image)/torch.linalg.norm(train_data[i,:]))
                            all_train_data.append(augmented_image)
                    #     if i < 12:
                    #         ax = axs.flat[i]
                    #         ax.matshow(torch.reshape(augmented_image, image_size), cmap='gray')
                    #         ax.axis('off')
                    #         ax.set_title(str(i))
                    # plt.show()

                train_data = torch.cat(all_train_data,0)
                # train_labels = train_labels.repeat(expand_fac+1)
                train_labels = torch.cat((train_labels, torch.unsqueeze(train_labels,1).repeat(expand_fac,hflip+rot+scal).reshape(train_labels.shape[0]*expand_fac*(hflip+rot+scal))), 0)

                print("number of training data is", train_data.shape[0])

                # pass

                # fig, axs = plt.subplots(2, 6, subplot_kw={'xticks': [], 'yticks': []})
                # for ax, idx in zip(axs.flat, torch.randperm(1000).detach().tolist()[:12]):
                #     one_atom = torch.reshape(train_data[idx, :], (256, 256)).detach().numpy()
                #     ax.matshow(one_atom, cmap='gray')
                #     ax.axis('off')
                #     ax.set_title(str(idx))
                # # plt.tight_layout()
                # plt.show()
                #
                # quit()

                # ###
                # # feature extraction
                # std_scaler = StandardScaler()
                # scaled_data = std_scaler.fit_transform(data) # use train data or all data?
                #
                # pca = PCA(n_components=feature_dim)
                # pca.fit_transform(scaled_data)
                # print("recovered covariance from PCA =",sum(pca.explained_variance_ratio_))
                #
                # data = torch.from_numpy(pca.transform(data).astype('float32'))
                # train_data = torch.from_numpy(pca.transform(train_data).astype('float32'))
                # test_data = torch.from_numpy(pca.transform(test_data).astype('float32'))
                # original_train_data = train_data
                # ###

            # aug_rep_accuracy = []
            # ini_dict = torch.tensor(0)
            # for i_out in range(aug_rep):
                # dictionary learning
                # initialize the dictionary (and other parameters)
                net = model.KDS(**hyp)
                if baseline == "no":
                    with torch.no_grad():
                        # if not consistent_dic.size() == torch.Size([]): ### temporary
                        #     net.W.data = consistent_dic ### temporary
                        # else: ### temporary
                        init_train_data = []
                        init_train_labels = []
                        if init_data == "original":
                            init_train_labels = original_train_labels
                            init_train_data = original_train_data
                        elif init_data == "all":
                            init_train_labels = train_labels
                            init_train_data = train_data
                        if init == "random_subset":
                            # if i_out == 0:
                            counts = torch.bincount(init_train_labels)
                            total_counts = len(init_train_data)
                            outputs = []
                            for idx in range(num_of_people):
                                count = counts[idx];
                                p = torch.randperm(count)
                                p = p[:int(net.hidden_size/num_of_people)]
                                outputs.append(init_train_data[init_train_labels == idx,:][p,:])
                            net.W.data = torch.cat(outputs, dim = 0)
                            # consistent_dic = net.W.data ### temporary
                                # ini_dict = net.W.data
                            # else:
                            #     net.W.data = ini_dict
                        elif init == "kmeans":
                            if config_num == 0: ### temporary
                                kmeans = KMeans(n_clusters=net.hidden_size).fit(init_train_data.detach().numpy())
                                net.W.data = torch.tensor(kmeans.cluster_centers_)
                                consistent_dic = net.W.data ### temporary
                            else: ### temporary
                                net.W.data = consistent_dic
                        # p = torch.randperm(len(train_data))[: net.hidden_size]
                        # net.W.data = train_data[p]
                        # net.W.data = train_data # seems to be some initialization. Or is this all dictionary? what about the net.hidden_size limit?
                        net.step.fill_((net.W.data.svd()[1][0] ** -2).item()) # based on my previous notes, this seems to be some step size that will be needed in the future.
                    net = net.to(device)
                    # net.hidden_size.fill_(net.W.data.shape[0]) # the hidden_size now becomes the number of training data.

                    if reg == "ex1":
                        if CVX_solver == False:
                            start_time = time.time()

                            # x_train_comp[config_num] = net.W.data ### temporary

                            # for now, I'm kind of treating this as a black box.
                            # All I know is after these operations,
                            # A and X are optimal according to our criterion
                            optimizer = optim.Adam(net.parameters(), lr=lr) # this step size is for the dictionary learning. If we are doing gradient descent, this is the step size.

                            # ### for the autograd approach
                            # x_hat = torch.zeros(train_data.size[0], hyp["hidden_size"], requires_grad=True)
                            # optimizer_direct = optim.Adam([x_hat], lr=lr)
                            # ###

                            criterion = utils.LocalDictionaryLoss(hyp["penalty"]) # this is an instance of a subclass of nn.Module. defines to be nn.Module because torch likes that.

                            net.train()
                            # if compare:
                            if dic_learning:
                                loss_arr = np.zeros(epochs)
                                for epoch in progress_bar(range(epochs)):
                                    # shuffle = torch.randperm(len(train_data)) ### temporarily commented out
                                    # train_data, train_labels = train_data[shuffle], train_labels[shuffle] ### temporarily commented out
                                    for i in progress_bar(range(0, len(train_data), batch_size), disable=True):
                                        y = train_data[i : i + batch_size].to(device)
                                        x_hat = net.encode(y)
                                        loss = criterion(net.W, y, x_hat)
                                        if epoch == 0:
                                            print("Initial value of the loss function: %.5f" % loss)
                                        if exact_dic == True:
                                            net.W.data = torch.linalg.solve((x_hat.T @ x_hat + 2*hyp["penalty"]*torch.diag(torch.sum(x_hat,0))).T, ((1+2*hyp["penalty"]) * y.T @ x_hat).T) # this is the exact solution of A given any X
                                        else:
                                            loss = criterion(net.W, y, x_hat)
                                            optimizer.zero_grad()
                                            loss.backward() # this calculate the gradient of respective variables
                                            nn.utils.clip_grad_norm_(net.parameters(), 1e-4)
                                            optimizer.step()
                                    loss = criterion(net.W, y, x_hat)
                                    loss_arr[epoch] = loss
                                    if epoch > 0:
                                        update_rate = (loss_arr[epoch - 1] - loss_arr[epoch]) / loss_arr[epoch - 1]
                                        print("Rate of update: %.9f" % update_rate)
                                        if 1e-6 > update_rate >= 0:
                                            break
                            elif not dic_learning:
                                shuffle = torch.randperm(len(train_data))
                                train_data, train_labels = train_data[shuffle], train_labels[shuffle]
                                for i in range(0, len(train_data), batch_size): # why do we need this at all? do not need this.
                                    y = train_data[i: i + batch_size].to(device)
                                    x_hat = net.encode(y)

                                # ### for the autograd approach
                                # y = train_data.to(device)
                                # loss = criterion(net.W, y, x_hat)
                                # for i in range(hyp["num_layers"]):
                                #     loss.backward()
                                #     optimizer_direct.step()
                                # ###

                            net.num_layers = torch.tensor(200)
                            # I just want to get the X for training data
                            with torch.no_grad():
                                net.eval() # use the latest dictionary
                                x_train_hat = []
                                for i in range(0, len(train_data)):
                                    y = train_data[[i]].to(device)
                                    x_train_hat.append(net.encode(y).cpu())
                                x_train_hat = torch.cat(x_train_hat)
                                loss = criterion(net.W, train_data, x_train_hat)
                                print("Final value of the loss function for training data: %.5f" % loss)

                            ###
                            # clf = LinearSVC().fit(x_train_hat.detach().numpy(),
                            #                       train_labels.detach().numpy())
                            # # preds = torch.tensor(clf.predict(test_data.detach().numpy()))
                            ###
                            # Actually assigning the labels
                            with torch.no_grad():
                                net.eval()
                                x_test_hat = []
                                preds = []
                                for i in progress_bar(range(0, len(test_data))):
                                    y = test_data[[i]].to(device)
                                    # TODO: randomly corrupt image here! This is a previous TODO.
                                    x_test_hat.append(net.encode(y).cpu())
                                    candidates = []
                                    x_test_hat_curr = x_test_hat[-1][0]
                                    # ###
                                    # preds.append(clf.predict(x_test_hat_curr.detach().numpy().reshape(1,-1))[0])
                                    # ###
                                    ### temporary
                                    # x_test_hat_curr[torch.nonzero(x_test_hat_curr, as_tuple=True)] = 1
                                    for j in range(k):
                                        x_train_hat_copy = x_train_hat.clone().detach()
                                        x_train_hat_copy = x_train_hat_copy[train_labels == j,:]
                                        # x_train_hat_copy[torch.nonzero(x_train_hat_copy, as_tuple=True)] = 1
                                        x_diff = x_test_hat_curr - x_train_hat_copy
                                        # matching_rate = 1 - torch.count_nonzero(x_diff)/torch.numel(x_diff)
                                        # matching_rate = 1 - torch.linalg.vector(x_diff,ord=1)/torch.norm(x_train_hat_copy,ord=1)
                                        size_train = np.shape(x_train_hat_copy)
                                        # print(size_train)
                                        # matching_rate = torch.linalg.matrix_norm(x_diff,ord=1)/size_train[0]
                                        # matching_rate = torch.linalg.vector_norm(x_diff,ord=1)/size_train[0]
                                        # print(size_train[0])
                                        # print(np.shape(x_test_hat_curr))
                                        size_test = np.shape(x_test_hat_curr)
                                        x_test_hat_curr_matrix = torch.ones(size_train[0],1)@torch.reshape(x_test_hat_curr,(1,size_test[0]))
                                        # print(np.shape(x_test_hat_curr_matrix))
                                        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                                        # print(np.shape(x_train_hat_copy))
                                        output = cos(x_test_hat_curr_matrix, x_train_hat_copy) # this computes the cosine similarity between the testing data and training data in this class. It's a vector
                                        # print(np.shape(output))
                                        output = torch.mean(cos(x_test_hat_curr_matrix, x_train_hat_copy)) # the "overall/average" cosine similarity between this testing data point and this class.
                                        # print(output)
                                        # print(np.shape(output))
                                        # quit()


                                        # candidates.append(matching_rate)
                                        candidates.append(output)

                                    # determine the prediction
                                    preds.append(np.argmax(candidates))
                                    # quality[i_out] += np.max(candidates) # criteria 1
                                    quality[i_out] += np.max(candidates) - np.sort(candidates)[-2] # criteria 2
                                    # print(preds[-1])
                                    # print(candidates)
                                    # print(np.max(candidates))
                                    # print(test_labels[i])

                                    # preds.append(np.argmax([x_hat[-1][0][train_labels == i].sum() for i in range(k)])) # this is appending a number
                                    ### temporary
                                    quality[i_out] += 1 ### temporary
                                x_test_hat = torch.cat(x_test_hat)
                                preds = torch.tensor(preds)

                                x_test_comp[config_num] = x_test_hat ### temporary
                                x_train_comp[config_num] = x_train_hat ### temporary

                                elapsed_time = (time.time()-start_time)
                                print("neural network took %.4f seconds to run" % elapsed_time)
                        elif CVX_solver == True:
                            start_time = time.time()
                            def loss_fn(A, X, Y):
                                return 1 / 2 * cp.norm(Y - X @ A, 'fro') ** 2
                            def regularizer(A, X, Y):
                                weight = np.sum(np.power((np.expand_dims(Y, axis = 1) - np.expand_dims(A, axis = 0)), 2), axis = 2)
                                obj = cp.sum(cp.sum(cp.multiply(weight, X), axis = 1))
                                return obj
                            def objective_fn(A, X, Y, penalty):
                                return loss_fn(A,X,Y) + penalty * regularizer(A,X,Y)

                            def QP_fn (A, X, Y, penalty):
                                weight = np.sum(np.power((np.expand_dims(Y, axis=1) - np.expand_dims(A, axis=0)), 2), axis=2)
                                P = A @ A.T
                                P = cp.psd_wrap(ss.kron(ss.eye(Y.shape[0]), P))
                                q = - A @ Y.T + penalty * weight.T
                                return (1/2) * cp.quad_form(cp.vec(X.T), P) + cp.trace(q.T @ X.T)

                            criterion = utils.LocalDictionaryLoss(hyp["penalty"])
                            if not dic_learning:
                                A = net.W.data.numpy()
                                print(A) ### temporary
                                Y_train = train_data.numpy()
                                print(Y_train) ### temporary
                                X_train = cp.Variable((np.shape(Y_train)[0],np.shape(A)[0]))

                                lambd = cp.Parameter(nonneg=True)

                                constraints_train = [X_train >= 0, cp.sum(X_train, axis=1) == 1]

                                problem_train = cp.Problem(cp.Minimize(QP_fn(A, X_train, Y_train, lambd)), constraints_train)

                                lambd.value = hyp["penalty"]

                                problem_train.solve()

                                if not problem_train.status == cp.OPTIMAL:
                                    CVX_solve_flag = False

                                y = train_data.to(device)
                                x_hat = torch.tensor(X_train.value, dtype=torch.float)
                                loss = criterion(net.W, y, x_hat)
                                print("Final value of the loss function for training data: %.5f" % loss)

                            else:
                                X_old = None
                                epochs_CVX = int(epochs/10)
                                loss_arr = np.zeros(epochs_CVX)
                                for epoch in progress_bar(range(epochs_CVX)):
                                    for i in progress_bar(range(0, len(train_data), batch_size), disable=True):
                                        A = net.W.data.numpy()
                                        Y_train = train_data[i: i + batch_size].numpy()
                                        # Y_train = train_data.numpy()
                                        # Y_test = test_data.data.numpy()
                                        # X_train = cp.Variable((np.shape(Y_train)[0],np.shape(A)[0]))
                                        # X_test = cp.Variable((np.shape(Y_test)[0],np.shape(A)[0]))

                                        X_train = cp.Variable((np.shape(Y_train)[0],np.shape(A)[0]), value = X_old)

                                        lambd = cp.Parameter(nonneg=True)

                                        constraints_train = [X_train >= 0, cp.sum(X_train, axis = 1) == 1]
                                        # constraints_test = [X_test >= 0, cp.sum(X_test, axis = 1) == 1]

                                        problem_train = cp.Problem(cp.Minimize(QP_fn(A, X_train, Y_train, lambd)), constraints_train)
                                        # problem_test = cp.Problem(cp.Minimize(QP_fn(A, X_test, Y_test, lambd)), constraints_test)

                                        lambd.value = hyp["penalty"]

                                        # if not X_old == None:
                                        #     problem_train.solve()

                                        problem_train.solve()
                                        # problem_test.solve()

                                        # print("number of iterations to solve the problem: ", problem_train.solver_stats.num_iters)

                                        # print(np.linalg.norm(X_train.value - X_train_ori.value, 'fro') / np.linalg.norm(X_train_ori.value, 'fro'))  ### temporary
                                        if not problem_train.status == cp.OPTIMAL:
                                            CVX_solve_flag = False

                                        x_hat = torch.tensor(X_train.value,dtype=torch.float)
                                        # X_old = X_train.value

                                        # solve for A
                                        if exact_dic:
                                            # need modification
                                            net.W.data = torch.linalg.solve(
                                                (x_hat.T @ x_hat + 2 * hyp["penalty"] * torch.diag(torch.sum(x_hat, 0))).T, ((1 + 2 * hyp["penalty"]) * torch.tensor(Y_train, dtype=torch.float).T @ x_hat).T)

                                        if epoch == epochs - 1:
                                            A = net.W.data.numpy()
                                            problem_train_final = cp.Problem(cp.Minimize(QP_fn(A, X_train, Y_train, lambd)),
                                                                       constraints_train)
                                            problem_train_final.solve()
                                    y = train_data.to(device)
                                    loss = criterion(net.W, y, x_hat)
                                    loss_arr[epoch] = loss
                                    if epoch == 0:
                                        print("Initial value of the loss function: %.5f" % loss)
                                    if epoch > 0:
                                        update_rate = (loss_arr[epoch - 1] - loss_arr[epoch]) / loss_arr[epoch - 1]
                                        # print("Rate of update: %.9f" % update_rate)
                                        if 1e-6 > update_rate >= 0:
                                            A = net.W.data.numpy()
                                            problem_train_final = cp.Problem(cp.Minimize(QP_fn(A, X_train, Y_train, lambd)),
                                                                       constraints_train)
                                            problem_train_final.solve()
                                            print("Final value of the loss function for training data: %.5f" % loss)
                                            break
                                    if epoch == epochs_CVX - 1:
                                        print("Final value of the loss function for training data: %.5f" % loss)

                            x_train_hat = torch.tensor(X_train.value)
                            x_hat = x_train_hat.detach().clone() # the saving the network step wants an x_hat

                            A = net.W.data.numpy()
                            Y_test = test_data.data.numpy()
                            X_test = cp.Variable((np.shape(Y_test)[0],np.shape(A)[0]))

                            lambd = cp.Parameter(nonneg=True)

                            constraints_test = [X_test >= 0, cp.sum(X_test, axis = 1) == 1]

                            problem_test = cp.Problem(cp.Minimize(QP_fn(A, X_test, Y_test, lambd)), constraints_test)

                            lambd.value = hyp["penalty"]

                            problem_test.solve()

                            if not problem_test.status == cp.OPTIMAL:
                                CVX_solve_flag = False

                            x_test_hat = torch.tensor(X_test.value)

                            preds = []
                            for i in progress_bar(range(0, len(test_data))):
                                # x_test_hat.append(torch.tensor(X_test.value[i,:]))
                                candidates = []
                                # x_test_hat_curr = x_test_hat[-1][0]
                                x_test_hat_curr = x_test_hat[i,:]
                                for j in range(k):
                                    # all relevant variables are tensors because there is a built-in CosineSimilarity in PyTorch
                                    x_train_hat_copy = x_train_hat.clone().detach()

                                    x_train_hat_copy = x_train_hat_copy[train_labels == j, :]
                                    size_train = np.shape(x_train_hat_copy)
                                    size_test = np.shape(x_test_hat_curr)
                                    x_test_hat_curr_matrix = torch.ones(size_train[0], 1).to(torch.float64) @ torch.reshape(x_test_hat_curr,
                                                                                                          (1, size_test[0])).to(torch.float64)
                                    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                                    output = cos(x_test_hat_curr_matrix,
                                                 x_train_hat_copy)  # this computes the cosine similarity between the testing data and training data in this class. It's a vector
                                    output = torch.mean(cos(x_test_hat_curr_matrix,
                                                            x_train_hat_copy))  # the "overall/average" cosine similarity between this testing data point and this class.
                                    candidates.append(output)

                                # determine the prediction
                                preds.append(np.argmax(candidates))
                                # quality[i_out] += np.max(candidates) # criteria 1
                                quality[i_out] += np.max(candidates) - np.sort(candidates)[-2]  # criteria 2

                            # x_test_hat = torch.cat(x_test_hat)
                            preds = torch.tensor(preds)

                            x_test_comp[config_num] = x_test_hat  ### temporary
                            x_train_comp[config_num] = x_train_hat ### temporary

                            elapsed_time = (time.time() - start_time)
                            print("CVX solver took %.4f seconds to run" % elapsed_time)
                    elif reg == "ex2":
                        start_time = time.time()
                        if not dic_learning:
                            A = net.W.data.numpy()
                            Y_train = train_data.numpy()
                            n = np.shape(Y_train)[0]
                            m = np.shape(A)[0]
                            d = np.shape(Y_train)[1]
                            X_train = np.zeros([n,m])
                            # may be able to do this in one go
                            for data_idx in range(n):
                                y = Y_train[data_idx,:]
                                C = A @ A.T + 2 * hyp["penalty"] * np.diag(np.sum(np.power((y - A),2),axis = 1))
                                C_inv = np.linalg.inv(C)
                                B = C_inv @ A @ y.T
                                X_train[data_idx,:] = C_inv @ (A @ y.T - ((y@A.T@C_inv@np.ones([m,1]) - 1)/np.sum(C_inv))*np.ones(m))
                        elif dic_learning:
                            epochs_ex2 = int(epochs / 10)

                        x_train_hat = torch.tensor(X_train)
                        x_hat = x_train_hat.detach().clone()  # the saving the network step wants an x_hat

                        A = net.W.data.numpy()
                        Y_test = test_data.numpy()
                        X_test = np.zeros([np.shape(Y_test)[0], np.shape(A)[0]])

                        for data_idx in range(np.shape(Y_test)[0]):
                            y = Y_test[data_idx, :]
                            C = A @ A.T + 2 * hyp["penalty"] * np.diag(np.sum(np.power((y - A), 2), axis=1))
                            C_inv = np.linalg.inv(C)
                            B = C_inv @ A @ y.T
                            X_test[data_idx, :] = C_inv @ (
                                        A @ y.T - ((y @ A.T @ C_inv @ np.ones([m, 1]) - 1) / np.sum(C_inv)) * np.ones(m))

                        x_test_hat = torch.tensor(X_test)

                        preds = []
                        for i in progress_bar(range(0, len(test_data))):
                            # x_test_hat.append(torch.tensor(X_test.value[i,:]))
                            candidates = []
                            # x_test_hat_curr = x_test_hat[-1][0]
                            x_test_hat_curr = x_test_hat[i, :]
                            for j in range(k):
                                # all relevant variables are tensors because there is a built-in CosineSimilarity in PyTorch
                                x_train_hat_copy = x_train_hat.clone().detach()

                                x_train_hat_copy = x_train_hat_copy[train_labels == j, :]
                                size_train = np.shape(x_train_hat_copy)
                                size_test = np.shape(x_test_hat_curr)
                                x_test_hat_curr_matrix = torch.ones(size_train[0], 1).to(torch.float64) @ torch.reshape(
                                    x_test_hat_curr,
                                    (1, size_test[0])).to(torch.float64)
                                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                                output = cos(x_test_hat_curr_matrix,
                                             x_train_hat_copy)  # this computes the cosine similarity between the testing data and training data in this class. It's a vector
                                output = torch.mean(cos(x_test_hat_curr_matrix,
                                                        x_train_hat_copy))  # the "overall/average" cosine similarity between this testing data point and this class.
                                candidates.append(output)

                            # determine the prediction
                            preds.append(np.argmax(candidates))
                            # quality[i_out] += np.max(candidates) # criteria 1
                            quality[i_out] += np.max(candidates) - np.sort(candidates)[-2]  # criteria 2

                        # x_test_hat = torch.cat(x_test_hat)
                        preds = torch.tensor(preds)

                        x_test_comp[config_num] = x_test_hat  ### temporary
                        x_train_comp[config_num] = x_train_hat  ### temporary

                        elapsed_time = (time.time() - start_time)
                        print("solving the ex2 problem took %.4f seconds to run" % elapsed_time)
                    ###
                    elif reg == "abs":
                        start_time = time.time()

                        def loss_fn(A, X, Y):
                            return 1 / 2 * cp.norm(Y - X @ A, 'fro') ** 2

                        def regularizer(A, X, Y):
                            weight = np.sum(np.power((np.expand_dims(Y, axis=1) - np.expand_dims(A, axis=0)), 2), axis=2)
                            obj = cp.sum(cp.sum(cp.multiply(weight, cp.abs(X)), axis=1))
                            return obj

                        def objective_fn(A, X, Y, penalty):
                            return loss_fn(A, X, Y) + penalty * regularizer(A, X, Y)


                        criterion = utils.LocalDictionaryLoss(hyp["penalty"])

                        if not dic_learning:
                            A = net.W.data.numpy()
                            print(A) ### temporary
                            Y_train = train_data.numpy()
                            print(Y_train) ### temporary
                            X_train = cp.Variable((np.shape(Y_train)[0], np.shape(A)[0]))

                            lambd = cp.Parameter(nonneg=True)

                            constraints_train = [cp.sum(X_train, axis=1) == 1]

                            problem_train = cp.Problem(cp.Minimize(objective_fn(A, X_train, Y_train, lambd)), constraints_train)

                            lambd.value = hyp["penalty"]

                            problem_train.solve()

                            if not problem_train.status == cp.OPTIMAL:
                                CVX_solve_flag = False

                            y = train_data.to(device)
                            x_hat = torch.tensor(X_train.value, dtype=torch.float)
                            loss = criterion(net.W, y, x_hat)
                            print("Final value of the loss function for training data: %.5f" % loss)

                        else:
                            X_old = None
                            epochs_CVX = int(epochs / 10)

                        x_train_hat = torch.tensor(X_train.value)
                        x_hat = x_train_hat.detach().clone()  # the saving the network step wants an x_hat

                        A = net.W.data.numpy()
                        Y_test = test_data.data.numpy()
                        X_test = cp.Variable((np.shape(Y_test)[0], np.shape(A)[0]))

                        lambd = cp.Parameter(nonneg=True)

                        constraints_test = [cp.sum(X_test, axis=1) == 1]

                        problem_test = cp.Problem(cp.Minimize(objective_fn(A, X_test, Y_test, lambd)), constraints_test)

                        lambd.value = hyp["penalty"]

                        problem_test.solve()

                        if not problem_test.status == cp.OPTIMAL:
                            CVX_solve_flag = False

                        x_test_hat = torch.tensor(X_test.value)

                        preds = []
                        for i in progress_bar(range(0, len(test_data))):
                            # x_test_hat.append(torch.tensor(X_test.value[i,:]))
                            candidates = []
                            # x_test_hat_curr = x_test_hat[-1][0]
                            x_test_hat_curr = x_test_hat[i, :]
                            for j in range(k):
                                # all relevant variables are tensors because there is a built-in CosineSimilarity in PyTorch
                                x_train_hat_copy = x_train_hat.clone().detach()

                                x_train_hat_copy = x_train_hat_copy[train_labels == j, :]
                                size_train = np.shape(x_train_hat_copy)
                                size_test = np.shape(x_test_hat_curr)
                                x_test_hat_curr_matrix = torch.ones(size_train[0], 1).to(torch.float64) @ torch.reshape(
                                    x_test_hat_curr,
                                    (1, size_test[0])).to(torch.float64)
                                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                                output = cos(x_test_hat_curr_matrix,
                                             x_train_hat_copy)  # this computes the cosine similarity between the testing data and training data in this class. It's a vector
                                output = torch.mean(cos(x_test_hat_curr_matrix,
                                                        x_train_hat_copy))  # the "overall/average" cosine similarity between this testing data point and this class.
                                candidates.append(output)

                            # determine the prediction
                            preds.append(np.argmax(candidates))
                            # quality[i_out] += np.max(candidates) # criteria 1
                            quality[i_out] += np.max(candidates) - np.sort(candidates)[-2]  # criteria 2

                        # x_test_hat = torch.cat(x_test_hat)
                        preds = torch.tensor(preds)

                        x_test_comp[config_num] = x_test_hat  ### temporary
                        x_train_comp[config_num] = x_train_hat  ### temporary

                        elapsed_time = (time.time() - start_time)
                        print("CVX solver took %.4f seconds to run" % elapsed_time)
                    ###
                elif baseline == "KM":
                    start_time = time.time()

                    criterion = utils.LocalDictionaryLoss(hyp["penalty"])

                    kmeans = KMeans(n_clusters=net.hidden_size).fit(original_train_data.detach().numpy())
                    col = kmeans.labels_
                    row = np.arange(original_train_data.shape[0])
                    content = np.ones(original_train_data.shape[0])
                    # x_train_hat = torch.tensor(ss.csr_matrix((content,(row,col)), shape = (original_train_data.shape[0],net.hidden_size)))
                    x_train_hat = ss.csr_matrix((content, (row, col)), shape=(original_train_data.shape[0], net.hidden_size))
                    print(x_train_hat)
                    x_train_hat = torch.tensor(x_train_hat.toarray())
                    x_hat = x_train_hat.detach().clone()  # the saving the network step wants an x_hat

                    col = kmeans.predict(test_data.data.numpy())
                    row = np.arange(test_data.shape[0])
                    content = np.ones(test_data.shape[0])
                    x_test_hat = torch.tensor(ss.csr_matrix((content,(row,col)), shape = (test_data.shape[0],net.hidden_size)).toarray())

                    # x_train_hat = torch.tensor(original_train_data.detach().numpy())
                    # # x_train_hat = torch.tensor(train_data.detach().numpy())
                    # x_hat = x_train_hat.detach().clone()  # the saving the network step wants an x_hat
                    # x_test_hat = torch.tensor(test_data.detach().numpy())

                    net.W.data = torch.tensor(kmeans.labels_)
                    consistent_dic = net.W.data  ### temporary

                    clf = LinearSVC().fit(x_train_hat.detach().numpy(),
                                          original_train_labels.detach().numpy())
                    preds = torch.tensor(clf.predict(x_test_hat.detach().numpy()))
                    quality[i_out] = 1

                    # preds = []
                    # for i in progress_bar(range(0, len(test_data))):
                    #     # x_test_hat.append(torch.tensor(X_test.value[i,:]))
                    #     candidates = []
                    #     # x_test_hat_curr = x_test_hat[-1][0]
                    #     x_test_hat_curr = x_test_hat[i, :]
                    #     for j in range(k):
                    #         # all relevant variables are tensors because there is a built-in CosineSimilarity in PyTorch
                    #         x_train_hat_copy = x_train_hat.clone().detach()
                    #
                    #         x_train_hat_copy = x_train_hat_copy[train_labels == j, :]
                    #         size_train = np.shape(x_train_hat_copy)
                    #         size_test = np.shape(x_test_hat_curr)
                    #         x_test_hat_curr_matrix = torch.ones(size_train[0], 1).to(torch.float64) @ torch.reshape(
                    #             x_test_hat_curr,
                    #             (1, size_test[0])).to(torch.float64)
                    #         cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                    #         output = cos(x_test_hat_curr_matrix,
                    #                      x_train_hat_copy)  # this computes the cosine similarity between the testing data and training data in this class. It's a vector
                    #         output = torch.mean(cos(x_test_hat_curr_matrix,
                    #                                 x_train_hat_copy))  # the "overall/average" cosine similarity between this testing data point and this class.
                    #         candidates.append(output)
                    #
                    #     # determine the prediction
                    #     preds.append(np.argmax(candidates))
                    #     # quality[i_out] += np.max(candidates) # criteria 1
                    #     quality[i_out] += np.max(candidates) - np.sort(candidates)[-2]  # criteria 2

                    # x_test_hat = torch.cat(x_test_hat)
                    preds = torch.tensor(preds)

                    x_test_comp[config_num] = x_test_hat  ### temporary
                    x_train_comp[config_num] = x_train_hat  ### temporary

                    elapsed_time = (time.time() - start_time)
                    print("KM took %.4f seconds to run" % elapsed_time)

                elif baseline == "SVM":
                    x_train_hat = torch.tensor(original_train_data.detach().numpy())
                    x_hat = x_train_hat.detach().clone()  # the saving the network step wants an x_hat

                    clf = LinearSVC().fit(original_train_data.detach().numpy(), original_train_labels.detach().numpy())
                    preds = torch.tensor(clf.predict(test_data.detach().numpy()))

                aug_rep_accuracy.append((test_labels == preds).float().mean().item())
                print(aug_rep_accuracy)

            # print(aug_rep_accuracy)
            # print("accuracy difference between 3nd and -3nd between initialization for one data set =", np.sort(aug_rep_accuracy)[-3] - np.sort(aug_rep_accuracy)[2])
            # print("maximum accuracy difference between initializations for one data set =", max(aug_rep_accuracy)-min(aug_rep_accuracy))
            # print("stdev. of accuracies =", np.std(aug_rep_accuracy))

            # combined_accuracy.append((test_labels == preds).float().mean().item())
            combined_accuracy[exp] = np.append(combined_accuracy[exp],max(aug_rep_accuracy)) # max from ground truth
            combined_accuracy_avg[exp] = np.append(combined_accuracy_avg[exp],mean(aug_rep_accuracy)) # average accuracy
            combined_accuracy_crit[exp] = np.append(combined_accuracy_crit[exp],aug_rep_accuracy[np.argmax(quality)]) # "max" from our criteria
            # print("how much worse is the accuracy from our criteria to the best possible accuracy", combined_accuracy[-1]-combined_accuracy_crit[-1])
            # print("how much better is the accuracy from our criteria to the average accuracy", combined_accuracy_crit[-1]-combined_accuracy_avg[-1])
            # print("accuracy =", (test_labels == preds).float().mean().item())

            # fig, axs = plt.subplots(2, 6, subplot_kw={'xticks': [], 'yticks': []})
            # for ax, idx in zip(axs.flat, range(12)):
            #     one_atom = torch.reshape(net.W[idx, :], image_size).detach().numpy()
            #     ax.matshow(one_atom, cmap='gray')
            #     ax.axis('off')
            #     ax.set_title(str(idx))
            # # plt.tight_layout()
            # plt.show()
            # # one_atom = torch.reshape(net.W[10,:], (192, 168)).detach().numpy()
            # # plt.matshow(one_atom, cmap='gray')
            # # plt.show()

        # print(torch.norm(x_test_comp[1] - x_test_comp[0], 'fro')) ### temporary index is hard coded.
        # ### temporary
        # AAA = x_train_comp[0]
        # AAB0 = ~np.isclose(AAA, np.zeros([train_data.shape[0],12]))
        # AAA[AAB0] >= 0
        # ### temporary
        # print("X_test percentage difference: %.5f" % (torch.norm(x_test_comp[1] - x_test_comp[0], 'fro') / torch.norm(x_test_comp[1], 'fro')))  ### temporary
        # print("X_train percentage difference: %.5f" % (torch.norm(x_train_comp[1] - x_train_comp[0], 'fro') / torch.norm(x_train_comp[1], 'fro')))  ### temporary

    # print results
    accuracy_diff = []
    accuracy_crit_diff = []
    accuracy_avg_diff = []
    combined_accuracy = np.array(combined_accuracy)
    combined_accuracy_crit = np.array(combined_accuracy_crit)
    combined_accuracy_avg = np.array(combined_accuracy_avg)
    if compare.shape[0] > 1:
        for comp_result_idx in list(combinations(list(range(compare.shape[0])), 2)): # make sure 1 is not a problem
        # for comp_result in list(combinations(compare,2)):
            comp_result = [compare[comp_result_idx[0]], compare[comp_result_idx[1]]]
            # comp_result_idx = [np.where(compare == comp_result[0]), np.where(compare == comp_result[1])]
            accuracy_diff.append(combined_accuracy[:,comp_result_idx[1]]-combined_accuracy[:,comp_result_idx[0]])
            accuracy_crit_diff.append(combined_accuracy_crit[:,comp_result_idx[1]]-combined_accuracy_crit[:,comp_result_idx[0]])
            accuracy_avg_diff.append(combined_accuracy_avg[:,comp_result_idx[1]]-combined_accuracy_avg[:,comp_result_idx[0]])

            text = "(accuracies from "
            text += utils.generate_text(comp_result[1], compare_poss)

            text += " - accuracies from "
            text += utils.generate_text(comp_result[0], compare_poss)
            text += ")"

            print("calculating results on", text)
            print("differences on best possible above accuracies =", accuracy_diff[-1])
            print("max of differences on best possible above accuracies =", np.max(accuracy_diff[-1]))
            print("min of differences on best possible above accuracies =", np.min(accuracy_diff[-1]))
            print("range of differences on best possible above accuracies = %.4f" % (np.max(accuracy_diff[-1]) - np.min(accuracy_diff[-1])))
            if exp_rep >= 5:
                print("range of differences on best possible above accuracies excluding top and bottom 20% results =", np.sort(accuracy_diff[-1])[-np.ceil(0.2*exp_rep).astype(int)-1] - np.sort(accuracy_diff[-1])[np.ceil(0.2*exp_rep).astype(int)])
            print("mean of differences on best possible above accuracies =", mean(accuracy_diff[-1]))
            print("median of differences on best possible above accuracies =", median(accuracy_diff[-1]))
            print("stdev. of differences on best possible above accuracies =", np.std(accuracy_diff[-1]))

            print("differences on best above accuracies from criteria =", accuracy_crit_diff[-1])
            print("max of differences on best above accuracies from criteria =", np.max(accuracy_crit_diff[-1]))
            print("min of differences on best above accuracies from criteria =", np.min(accuracy_crit_diff[-1]))
            print("range of differences on best above accuracies from criteria = %.4f" % (np.max(accuracy_crit_diff[-1]) - np.min(accuracy_crit_diff[-1])))
            if exp_rep >= 5:
                print("range of differences on best above accuracies from criteria excluding top and bottom 20% results =",
                      np.sort(accuracy_crit_diff[-1])[-np.ceil(0.2 * exp_rep).astype(int) - 1] - np.sort(accuracy_crit_diff[-1])[
                          np.ceil(0.2 * exp_rep).astype(int)])
            print("mean of differences on best above accuracies from criteria =", mean(accuracy_crit_diff[-1]))
            print("median of differences on best above accuracies from criteria =", median(accuracy_crit_diff[-1]))
            print("stdev. of differences on best above accuracies from criteria =", np.std(accuracy_crit_diff[-1]))

            print("differences on average above accuracies =", accuracy_avg_diff[-1])
            print("max of differences on average above accuracies =", np.max(accuracy_avg_diff[-1]))
            print("min of differences on average above accuracies =", np.min(accuracy_avg_diff[-1]))
            print("range of differences on average above accuracies = %.4f" % (np.max(accuracy_avg_diff[-1]) - np.min(accuracy_avg_diff[-1])))
            if exp_rep >= 5:
                print("range of differences on average above accuracies excluding top and bottom 20% results =",
                      np.sort(accuracy_avg_diff[-1])[-np.ceil(0.2 * exp_rep).astype(int) - 1] - np.sort(accuracy_avg_diff[-1])[
                          np.ceil(0.2 * exp_rep).astype(int)])
            print("mean of differences on average above accuracies =", mean(accuracy_avg_diff[-1]))
            print("median of differences on average above accuracies =", median(accuracy_avg_diff[-1]))
            print("stdev. of differences on average above accuracies =", np.std(accuracy_avg_diff[-1]))

            print("")

    for idx in range(compare.shape[0]):
        text = "accuracies from " + utils.generate_text(compare[idx], compare_poss)

        print("calculating results on", text)

        print("best possible accuracies =", combined_accuracy[:, idx])
        print("range of best possible accuracies = %.4f" % (np.max(combined_accuracy[:, idx]) - np.min(combined_accuracy[:, idx])))
        if exp_rep >= 5:
            print("range of best possible accuracies excluding top and bottom 20% results =", np.sort(combined_accuracy[:, idx])[-np.ceil(0.2*exp_rep).astype(int)-1] - np.sort(combined_accuracy[:, idx])[np.ceil(0.2*exp_rep).astype(int)])
        print("best possible mean accuracy =", mean(combined_accuracy[:, idx]))
        print("best possible median accuracy =", median(combined_accuracy[:, idx]))
        print("stdev. of best possible accuracies =", np.std(combined_accuracy[:, idx]))

        print("best accuracies from criteria =", combined_accuracy_crit[:, idx])
        print("range of best accuracies from criteria = %.4f" % (np.max(combined_accuracy_crit[:, idx]) - np.min(combined_accuracy_crit[:, idx])))
        if exp_rep >= 5:
            print("range of best accuracies from criteria excluding top and bottom 20% results =", np.sort(combined_accuracy_crit[:, idx])[-np.ceil(0.2*exp_rep).astype(int)-1] - np.sort(combined_accuracy_crit[:, idx])[np.ceil(0.2*exp_rep).astype(int)])
        print("mean of best accuracies from criteria =", mean(combined_accuracy_crit[:, idx]))
        print("median of best accuracies from criteria =", median(combined_accuracy_crit[:, idx]))
        print("stdev. of best accuracies from criteria =", np.std(combined_accuracy_crit[:, idx]))

        print("average accuracy =", combined_accuracy_avg[:, idx])
        print("range of average accuracy = %.4f" % (np.max(combined_accuracy_avg[:, idx]) - np.min(combined_accuracy_avg[:, idx])))
        if exp_rep >= 5:
            print("range of average accuracy excluding top and bottom 20% results =", np.sort(combined_accuracy_avg[:, idx])[-np.ceil(0.2*exp_rep).astype(int)-1] - np.sort(combined_accuracy_avg[:, idx])[np.ceil(0.2*exp_rep).astype(int)])
        print("mean of average accuracies =", mean(combined_accuracy_avg[:, idx]))
        print("median of average accuracies =", median(combined_accuracy_avg[:, idx]))
        print("stdev. of average accuracies =", np.std(combined_accuracy_avg[:, idx]))

        print("")

    if True in compare[:,compare_poss.index('CVX_solver')]:
        if CVX_solve_flag == True:
            print("all CVX problems are solved")
        else:
            print("not all CVX problems are solved")
        print("")



    print("saving network...")
    path = Path(_run.observers[0].dir if _run.observers else ".")
    save = {
        "net": net.state_dict(),
    }
    torch.save(save, path / "network_state.pt")

    scipy.io.savemat(
        path / "coefficient_matrix.mat",
        mdict = {
            "data": test_data.numpy(),
            "labels": test_labels.numpy(),
            "atoms": net.W.data.clone().detach().cpu().numpy(),
            "coefficients": x_hat.clone().detach().cpu().numpy(),
            "predictions": preds.clone().detach().cpu().numpy(),
        },
    )

    return
