import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
from utils import load_data, separate_data, GB_load_data, coarsening_load_data
from models.graphcnn import GraphCNN
from datetime import datetime

criterion = nn.CrossEntropyLoss()

def coarsening_graph(coarsening_method, coarsening_ratio, dataset, degree_as_tag, purity, degree_purity):
    """
    Load and coarsen the graph dataset based on the specified method and parameters.
    
    Args:
        coarsening_method (str): The method used for coarsening the graph.
        coarsening_ratio (float): The ratio for coarsening.
        dataset (str): The name of the dataset.
        degree_as_tag (bool): Whether to use node degree as a tag.
        purity (float): Purity criterion for coarsening.
        degree_purity (bool): Whether to use degree-based purity first.

    Returns:
        tuple: The coarsened graphs and number of classes.
    """
    if coarsening_method == "GBGC":
        graphs, num_classes = GB_load_data(dataset, degree_as_tag, purity, degree_purity)
    elif coarsening_method in ["vgc", "vegc", "mgc", "sgc", "wgc"]:
        graphs, num_classes = coarsening_load_data(dataset, coarsening_method, coarsening_ratio, degree_as_tag)
    else:
        graphs, num_classes = load_data(dataset, degree_as_tag)
    return graphs, num_classes

def main():
    """
    Main function to set up and train the graph convolutional neural network for whole-graph classification.
    """
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="MUTAG", help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50, help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0, help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0, help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=5, help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2, help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64, help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5, help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"], help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"], help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true", help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--degree_as_tag', action="store_true", help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type=str, default="", help='output file')
    parser.add_argument('--coarsening_method', type=str, default="origin", help='coarsening_method')
    parser.add_argument('--coarsening_ratio', type=float, default=0.2, help='coarsening_ratio')
    parser.add_argument('--purity', type=float, default=0, help='纯度使用：1. 0 为不使用 2. 2 为纯度自适应 3. 3 纯度加结构自适应 4. 0-1 为纯度设置')
    parser.add_argument('--degree_purity', type=str, default='False', help='True为先使用结构分裂，再使用纯度分裂')

    args = parser.parse_args()

    if args.degree_purity.lower() == 'true':
        args.degree_purity = True
    elif args.degree_purity.lower() == 'false':
        args.degree_purity = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed + 9)    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    print("torch.cuda.is_available() = ", torch.cuda.is_available())
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + 5)

    dataset_name = args.dataset 
    fold_idx_name = args.fold_idx
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = f'log/{dataset_name}_{fold_idx_name}_{current_time}'  
    writer = SummaryWriter(log_dir)

    print("data:", args.dataset, ", method:", args.coarsening_method, ", degree_purity:", args.degree_purity, ", purity:", args.purity, ", ratio:", args.coarsening_ratio, ", fold_idx:", args.fold_idx)
    print("batch_size:", args.batch_size, ", lr:", args.lr, ", seed:", args.seed, 
          ", num_layers:", args.num_layers, ", num_mlp_layers:", args.num_mlp_layers, ", graph_pooling_type:", args.graph_pooling_type, ', neighbor_pooling_type:', args.neighbor_pooling_type,
          ", hidden_dim:", args.hidden_dim, ", final_dropout:", args.final_dropout, "step_size=50, gamma=0.5")

    graphs, num_classes = coarsening_graph(args.coarsening_method, args.coarsening_ratio, args.dataset, args.degree_as_tag, args.purity, args.degree_purity)

    train_graphs, test_graphs = separate_data(graphs, 0, args.fold_idx)
    print("train_graphs", len(train_graphs), "test_graphs", len(test_graphs))
    start_time = time.time()
    max_test_accuracy = 0.0
    max_train_accuracy = 0.0

    model = GraphCNN(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type, device).to(device)
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print("== Total number of parameters: {}".format(num_params))
    num_params_update = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("== Total number of learning parameters: {}".format(num_params_update))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    for epoch in range(1, args.epochs + 1):
        avg_loss = train(args, model, device, train_graphs, test_graphs, optimizer, epoch, criterion)
        scheduler.step()
        acc_train, acc_test = test(args, model, device, train_graphs, test_graphs, epoch)
        if acc_test > max_test_accuracy:
            max_test_accuracy = acc_test
            max_train_accuracy = acc_train
            epoch_num = epoch
            print("max_train_accuracy: ", max_train_accuracy, " max_test_accuracy: ", max_test_accuracy)

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        train_loss, test_loss = avg_loss
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/train', acc_train, epoch)
        writer.add_scalar('Accuracy/test', acc_test, epoch)

        if not args.filename == "":
            with open(args.filename, 'w') as f:
                f.write("%f %f %f" % (avg_loss, acc_train, acc_test))
                f.write("\n")
        print("")
        print(model.eps)
        writer.close()

    end_time = time.time()    
    total_time = int(end_time - start_time)
    hours = total_time // 3600  
    minutes = (total_time % 3600) // 60  
    seconds = total_time % 60  
    print(f"Total training time: {hours} hours, {minutes} minutes, {seconds} seconds.")
    print("data:", args.dataset, ", method:", args.coarsening_method, " ,degree_purity:", args.degree_purity, " ,purity:", args.purity, ", ratio:", args.coarsening_ratio, ", fold_idx:", args.fold_idx)
    print(f"Maximum train accuracy: {max_train_accuracy:.4f}")
    print(f"Maximum test accuracy: {max_test_accuracy:.4f}")
    print("epoch:", epoch_num)

def train(args, model, device, train_graphs, test_graphs, optimizer, epoch, criterion):
    """
    Train the model for one epoch.
    
    Args:
        args (Namespace): The arguments parsed from the command line.
        model (nn.Module): The graph convolutional network model.
        device (torch.device): The device to train the model on.
        train_graphs (list): The training graph data.
        test_graphs (list): The test graph data.
        optimizer (Optimizer): The optimizer for training.
        epoch (int): The current epoch number.
        criterion (Loss): The loss function.

    Returns:
        tuple: The average training loss and average test loss.
    """
    model.train()
    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')
    loss_accum = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]
        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output = model(batch_graph)
        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)
        loss = criterion(output, labels)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss = loss.item()
        loss_accum += loss
        pbar.set_description('epoch: %d' % (epoch))
    average_loss_train = loss_accum / total_iters
    model.eval()
    with torch.no_grad():
        test_loss_accum = 0
        for graph in test_graphs:
            output = model([graph])
            label = torch.LongTensor([graph.label]).to(device)
            test_loss = criterion(output, label)
            test_loss_accum += test_loss.item()
        average_loss_test = test_loss_accum / len(test_graphs)
    current_lr = optimizer.param_groups[0]['lr']
    print("training loss: %f, test loss: %f, lr: %f" % (average_loss_train, average_loss_test, current_lr))
    return average_loss_train, average_loss_test

def pass_data_iteratively(model, graphs, minibatch_size=64):
    """
    Pass the data iteratively through the model to avoid memory overflow.
    
    Args:
        model (nn.Module): The graph convolutional network model.
        graphs (list): The graph data.
        minibatch_size (int): The size of each mini-batch.

    Returns:
        torch.Tensor: The concatenated output from the model.
    """
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)

def test(args, model, device, train_graphs, test_graphs, epoch):
    """
    Test the model and calculate the accuracy on the training and test datasets.
    
    Args:
        args (Namespace): The arguments parsed from the command line.
        model (nn.Module): The graph convolutional network model.
        device (torch.device): The device to test the model on.
        train_graphs (list): The training graph data.
        test_graphs (list): The test graph data.
        epoch (int): The current epoch number.

    Returns:
        tuple: The training accuracy and test accuracy.
    """
    model.eval()
    output = pass_data_iteratively(model, train_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    print("train acc: %f test acc: %f" % (acc_train, acc_test))

    return acc_train, acc_test    

if __name__ == '__main__':
    main()
