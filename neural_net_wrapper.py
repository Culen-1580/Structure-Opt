import os
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from graph_game import GraphGame
from dot_dict import DotDict
from average_meter import AverageMeter
from graph_neural_net import GraphNeuralNet

class NeuralNetWrapper():
  def __init__(self, args: DotDict, game: GraphGame) -> None:
    self.args = args
    self.device = args.device
    self.epochs = args.epochs
    self.batch_size = args.batch_size
    self.n_max_node = args.n_max_node
    self.n_feature = args.n_feature
    self.nnet = GraphNeuralNet(args, game)

  def train(self, samples: list):
    """
    samples: list of training data samples of format (state, pi, v)
    """
    batch_size = self.batch_size
    optimizer = optim.Adam(self.nnet.parameters())

    for e in range(self.epochs):
      # print info
      print(f'Epoch: {e + 1}, Num of Samples: {len(samples)}')
      self.nnet.train()
      pi_losses = AverageMeter()

      batch_count = int(len(samples) / batch_size)
      t = tqdm(range(batch_count), desc='Training Net')
      for _ in t:
        # TODO: need to exclude duplicates in sample_ids?
        sample_ids = np.random.randint(len(samples), size=batch_size)
        states, pis, vs = list(zip(*[samples[i] for i in sample_ids]))

        # hs_init, adjs0, adjs1, and adjs2 are in GPU
        hs_init, adjs0, adjs1, adjs2 = self.convertArrayToMatrix(states)
        # request contiguous memory and move to GPU
        target_pis = torch.tensor(np.array(pis, dtype=np.float32)).contiguous().to(self.device)
        # target_vs = torch.tensor(np.array(vs)).contiguous().to(self.device)

        # compute output: action probability vectors
        out_pis = self.nnet(hs_init, adjs0, adjs1, adjs2)

        # compute loss
        loss = self.loss_pi(target_pis, out_pis)

        # record loss
        pi_losses.update(loss.item(), batch_size)
        t.set_postfix(Loss=pi_losses.avg)

        # compute gradient and do a optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

  def predict(self, state: np.array):
    """
    state: represents a graph
    """
    # start time
    # start = time.time()

    # preparing input: set batch_size 1
    states = [state]
    hs_init, adjs0, adjs1, adjs2 = self.convertArrayToMatrix(states)

    # switch to inference/evaluating
    self.nnet.eval()
    with torch.no_grad():
      pis = self.nnet(hs_init, adjs0, adjs1, adjs2)

    # print('Prediction time taken: {0:03f}'.format(time.time()-start))
    # 'state', 'pi', and 'v' all have batch size 1
    # so using '[0]' to get the first element from the batch
    return pis.data.to('cpu').numpy()[0]

  def convertArrayToMatrix(self, states: list):
    # states: list of np.array, each has shape:
    # (node_num + (node_num - 1) x 2) x 2
    sample_num = len(states)

    hs_init = torch.zeros(sample_num, self.n_max_node, self.n_max_node, device=self.device, requires_grad=False)
    adjs0 = torch.zeros(sample_num, self.n_max_node, self.n_max_node, device=self.device, requires_grad=False)
    adjs1 = torch.zeros(sample_num, self.n_max_node, self.n_max_node, device=self.device, requires_grad=False)
    adjs2 = torch.zeros(sample_num, self.n_max_node, self.n_max_node, device=self.device, requires_grad=False)

    for i in range(sample_num):
      state = states[i]
      node_num = int((state.shape[0] + 2) / 3)
      assert(node_num * 3 - 2 == state.shape[0])

      for j in range(node_num):
        # degree = state[j][1] - state[j][0]
        for k in range(state[j][0], state[j][1]):
          neighbor = state[k][0]
          edge_type = state[k][1]
          if (edge_type == 0):
            adjs0[i][j][neighbor] = 1.0
          elif (edge_type == 1):
            adjs1[i][j][neighbor] = 1.0
          elif (edge_type == 2):
            adjs2[i][j][neighbor] = 1.0
        hs_init[i][j][j] = 1.0
    return hs_init, adjs0, adjs1, adjs2

  def loss_pi(self, targets: torch.tensor, outputs: torch.tensor):
    return torch.sum(targets * outputs) / targets.size()[0]

  def save_checkpoint(self,
                      folder='checkpoint',
                      filename='checkpoint.pth.tar'):
    filepath = os.path.join(folder, filename)
    if not os.path.exists(folder):
      print("No checkpoint directory! Make directory {}".format(folder))
      os.mkdir(folder)
    else:
      print("Checkpoint directory exists!")
    torch.save({'state_dict': self.nnet.state_dict(),}, filepath)

  def load_checkpoint(self,
                      folder='checkpoint',
                      filename='checkpoint.pth.tar'):
    filepath = os.path.join(folder, filename)
    if not os.path.exists(filepath):
      raise ("No model in path {}".format(filepath))
    map_location = None if (self.device == 'cuda:0') else 'cpu'
    checkpoint = torch.load(filepath, map_location=map_location)
    self.nnet.load_state_dict(checkpoint['state_dict'])
