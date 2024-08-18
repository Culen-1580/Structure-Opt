import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_game import GraphGame
from dot_dict import DotDict

class GraphNeuralNet(nn.Module):
  def __init__(self, args: DotDict, game: GraphGame):
    super(GraphNeuralNet, self).__init__()
    
    # parameters
    self.device = args.device
    self.dropout = args.dropout
    self.n_layer = args.n_layer
    self.n_feature = args.n_feature
    self.n_max_node = args.n_max_node
    self.batch_size = args.batch_size
    self.action_size = game.getActionSize()

    # initial feature transformation from one-hot vector to length-self.n_feature vector
    # one-hot vector: identity for each node
    self.W_init = torch.randn(self.n_max_node, self.n_feature, device=self.device)

    # GNN parameters
    # W0: incoming solid edges
    self.W0 = torch.randn(self.n_layer, self.n_feature, self.n_feature, device=self.device)
    # W1: incoming dash edges
    self.W1 = torch.randn(self.n_layer, self.n_feature, self.n_feature, device=self.device)
    # W2: incoming dash edges with -1
    self.W2 = torch.randn(self.n_layer, self.n_feature, self.n_feature, device=self.device)
    # W3: self feature transformation
    self.W3 = torch.randn(self.n_layer, self.n_feature, self.n_feature, device=self.device)
    # b: bias terms
    self.b = torch.randn(self.n_layer, self.n_feature, device=self.device)

    # full connection layers
    self.fc1 = nn.Linear(self.n_feature, self.action_size, device=self.device)

    # bactch normalize
    self.bn_init = nn.BatchNorm1d(self.n_max_node, device=self.device)
    self.bn_layer = [nn.BatchNorm1d(self.n_max_node, device=self.device) for _ in range(self.n_layer)]
    self.bn_graph = nn.BatchNorm1d(self.n_feature, device=self.device)

    # leaky relu
    # self.leaky_relu = nn.LeakyReLU(0.1)

  def forward(self, hs_init: torch.tensor, adjs0: torch.tensor, adjs1: torch.tensor, adjs2: torch.tensor):
    # hs_init (batch_size X n_max_node X n_max_node): 
    # initial feature vectors for each node, row i for node i (row-wise)
    # adjs0, adjs1, adjs2: (batch_size X n_max_node X n_max_node)
    # adjs0: solid edges, adjs1: dash edges, adjs2: dash edges with -1
    # adjacent matrix stores the incoming-edge neighbors, row i for node i (row-wise)

    # connections from all incoming edges
    # adjs: (batch_size X n_max_node X n_max_node)
    adjs = adjs0 + adjs1 + adjs2
    # conn, conn0, conn1, conn2: ((n_layer + 1) X batch_size X n_max_node X n_max_node)
    conn = [adjs]
    conn0 = [adjs0]
    conn1 = [adjs1]
    conn2 = [adjs2]

    # prepare input for the first layer
    # h: ((n_layer + 1) X batch_size X n_max_node X n_feature)
    h = [F.relu(self.bn_init(torch.matmul(hs_init, self.W_init)))]
    # h = [self.leaky_relu(self.bn_init(torch.matmul(hs_init, self.W_init)))]
    ### print(f'h[0][0][5]:\n{h[0][0][5]}')
    ### print(f'h[0][9][5]:\n{h[0][9][5]}')

    # compute results for each layer
    for l in range(self.n_layer):
      # for feature vectors at each layer, input and output have the same shape: (batch_size X n_max_node X n_feature)
      # b[l] (n_feature): bias will be added to each row
      h_tmp = torch.matmul(h[l], self.W3[l]) + self.b[l]
      for i in range(l + 1):
        # sum_h0, sum_h1, sum_h2: (batch_size X n_max_node X n_feature):
        # sum of neighbors' feature vectors for each node
        sum_h0 = torch.matmul(conn0[i], h[l - i])
        sum_h1 = torch.matmul(conn1[i], h[l - i])
        sum_h2 = torch.matmul(conn2[i], h[l - i])
        h_tmp += (torch.matmul(sum_h0, self.W0[l]) + torch.matmul(sum_h1, self.W1[l]) + torch.matmul(sum_h2, self.W2[l]))
      h.append(F.relu(self.bn_layer[l](h_tmp)))
      # h.append(F.leaky_relu(self.bn_layer[l](h_tmp)))
      ### print(f'h[{l+1}][0][5]:\n{h[l+1][0][5]}')
      ### print(f'h[{l+1}][9][5]:\n{h[l+1][9][5]}')

      conn.append(torch.matmul(conn[l], adjs))
      conn0.append(torch.matmul(conn[l], adjs0))
      conn1.append(torch.matmul(conn[l], adjs1))
      conn2.append(torch.matmul(conn[l], adjs2))

    # graph_h (batch_size X n_feature): graph feature vector
    sum_result = torch.sum(h[self.n_layer], dim=1)
    ### print(f'sum_result shape: \n{sum_result.shape}')
    ### print(f'sum_result: \n{sum_result}')

    out_feature = self.bn_graph(sum_result)
    ### print(f'out_feature shape: \n{out_feature.shape}')
    ### print(f'out_feature: \n{out_feature}')
    ### print(f'out_feature[0]: \n{out_feature[0]}')
    ### print(f'out_feature[9]: \n{out_feature[9]}')

    graph_h = F.dropout(F.relu(out_feature), p=self.dropout, training=self.training)
    # graph_h = F.dropout(self.leaky_relu(out_feature), p=self.dropout, training=self.training)
    ### print(f'graph_h shape: \n{graph_h.shape}')
    ### print(f'graph_h: \n{graph_h}')
    ### print(f'graph_h[0]: \n{graph_h[0]}')
    ### print(f'graph_h[9]: \n{graph_h[9]}')
    
    # predict prability vector and score value
    pis = torch.exp(F.log_softmax(self.fc1(graph_h), dim=1))
    return pis