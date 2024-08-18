import logging
import os
import sys
import copy
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
import numpy as np
import heapq as pq
import torch.multiprocessing as mp
from tqdm import tqdm
from queue import Queue

from neural_net_wrapper import NeuralNetWrapper as Net
from graph_game import GraphGame as Game
from arena import Arena
from mcts import MCTS
from dot_dict import DotDict

log = logging.getLogger(__name__)

class Train():
  """
  This class executes game playing and model training.
  """
  def __init__(self, args: DotDict, game: Game, nnet: Net) -> None:
    self.game = game
    self.args = args
    self.EPS = args.EPS
    self.n_max_node = args.n_max_node
    self.n_max_act = args.n_max_act
    self.n_iter = args.n_iter
    self.n_epsds = args.n_epsds
    self.n_steps = args.n_steps
    self.max_samples_per_iter = args.max_samples_per_iter
    self.n_arena_games = args.n_arena_games
    # current neural network
    self.curr_nnet = nnet
    # previous neural network
    self.prev_nnet = self.curr_nnet.__class__(args, game)
    # training data collected from all previous iterations
    # TODO: accumulated training data from the last 'args.max_iters_for_data' iterations
    self.accu_train_data = {}

  def executeEpisode(self, train_data: dict) -> None:
    """
    This function executes 1 episode/round of game.
    In this graph game, each episode has 'n_steps' steps.
    Each step corresponds a game state, a action prob vector, and a reward value,
    where action prob vector is computed by MCTS and reward corresponds to the min
    graph size among 'n_max_act' steps starting from current step.
    Only unique states and the associated data will be collected as training samples.
    For simplicity, only state and action prob vector will be used for training.
    Reward is used to choose the best prob vector for each unique state, but it
    is not used for training, i.e., not used to compute training loss.

    Returns 'train_data':
      dict of samples: key is order, data is (state, pi, v)
      order: unique representation of game state in string
      state: state of game at each step
      pi: action probability/policy vector returned from MCTS
      v: max reward of playing 'n_max_act' steps from current state
    """
    # initial graph
    game = copy.copy(self.game)
    # initialize Monte-Carlo search tree
    mcts = MCTS(self.curr_nnet, self.args)
    # queue stores data samples generated in training
    que = Queue(maxsize=(self.n_max_act + 1))
    # priority queue stores reward values
    rewards = []
    pq.heapify(rewards) # no need
    
    if (game.getGraphSize() > self.n_max_node):
      log.error("Intial graph size > n_max_node. Exit!")
      print("Intial graph size > n_max_node. Exit!")
      return []
    
    process_name = mp.current_process().name
    for step in tqdm(range(self.n_steps), desc=f'{process_name}, steps'):
      # execute MCTS from current state
      # When training, 'mode=1'; when inference, 'mode=0'
      order = game.getOrderByEdges()
      state = game.getStateInNpArray()
      v = game.getReward()
      pi = mcts.getActionProb(game, mode=1, step=step)

      ### # print info
      ### if ((step % self.args.sampling_dis) == 1):
      ###   print(f'\nstep: {step}  ')
      ###   print(f'  order: {order}')
      ###   # print(f'  state:\n{state}')
      ###   np.set_printoptions(precision=2)
      ###   print(f'  pi: {pi}')
      ###   print('  v: {:0.4f}'.format(v))
      ###   ### file_name = order + '.png'
      ###  ### game.dumpGraph(file_name)

      que.put((order, state, pi, v))
      # min heap, but want max value, so use '-v'
      pq.heappush(rewards, (-v, step))

      if (que.full()):
        (tmp_order, tmp_state, tmp_pi, _) = que.get()
        left = step - self.n_max_act
        while (rewards[0][1] < left):
          pq.heappop(rewards)
        # negate to get true reward value
        tmp_v = -rewards[0][0]
        # update the data sample in train_data
        if ((tmp_order not in train_data) or (train_data[tmp_order][2] <= tmp_v)):
          train_data[tmp_order] = (tmp_state, tmp_pi, tmp_v)

      # action prob vector should be non-empty, because can take action 0
      # except 'n_MCTS' == 1 or 'n_max_act' == 0
      sum_pi = np.sum(pi)
      assert(sum_pi > self.EPS)

      # take action to get next state
      moves = game.getValidActions()
      action = np.random.choice(len(pi), p=pi)
      while (moves[action] < self.EPS):
        action = np.random.choice(len(pi), p=pi)
        assert(False)
      game.getNextState(action)

  def learn(self):
    """
    Performs 'n_iter' iterations of playing games and training the neural net.
    Each iteration:
      play 'n_epsd' episodes/rounds of the game,
      add training data samples to 'accu_train_data',
      use accumulated training data to train the neural net,
      pit the new neural net against the old one,
      accept the new if its winning rate against the old >= 'accept_threshold'.
    """
    mp.set_start_method('spawn')
    # each iteration: play 'n_epsds' episodes/rounds of game to train and get a new neural model
    # accept the new model and replace the old, if the new is better 
    for k in range(1, self.n_iter + 1):
      log.info(f'Starting training iteration: {k}')

      # training data from current iteration
      # TODO: set dict size to 'self.max_samples_per_iter'
      iter_train_data = {}

      # collect training data for each episode
      manager = mp.Manager()
      epsd_train_data = [manager.dict() for _ in range(self.n_epsds)]

      # process list
      processes = []

      print(f'Play {self.n_epsds} rounds/episodes:')
      # each episode: play 'n_steps' steps of the game
      for i in range(self.n_epsds):
        # run episode in seperate processes to get training data
        name = 'process-' + str(i)
        p = mp.Process(target=self.executeEpisode, args=(epsd_train_data[i],), name=name)
        processes.append(p)
        p.start()
      # wait for all threads to finish
      for p in processes:
        p.join()

      print(f'All {self.n_epsds} threads finished! Now merge training data!')
      for i in range(self.n_epsds):
        collects = epsd_train_data[i]
        for order in collects:
          if ((order not in iter_train_data) or (iter_train_data[order][2] <= collects[order][2])):
            iter_train_data[order] = collects[order]

      # merge training data of current iteration to global training data
      for order in iter_train_data:
        if ((order not in self.accu_train_data) or (self.accu_train_data[order][2] <= iter_train_data[order][2])):
          self.accu_train_data[order] = iter_train_data[order]

      # print global training data to file
      fp = open('data.txt', 'a')
      fp.write(f'\niteration: {k}\n')
      fp.write(f'number of training samples accumulated: {len(self.accu_train_data)}\n')
      for item in self.accu_train_data:
        fp.write(f'  order: {item}\n')
        # print(f'  state:\n{self.accu_train_data[item][0]}')
        np.set_printoptions(precision=2)
        fp.write(f'  pi: {self.accu_train_data[item][1]}\n')
        fp.write(f'  v: {self.accu_train_data[item][2]}\n\n')
      fp.close()

      ### if len(self.accu_train_data) > self.args.max_iters_for_data:
      ###   log.warning(
      ###     'Delete training data of the oldest iteration. '
      ###     f'Size of accu_train_data: {len(self.accu_train_data)}')
      ###   self.accu_train_data.pop(0)
      
      ### # save accumulated training data into a file
      ### # model is trained using data from previous iterations 0, ..., i-1
      ### self.saveTrainData(i - 1)

      ### # shuffle data for later training
      ### train_data = []
      ### for e in self.accu_train_data:
      ###   train_data.extend(e)
      ### shuffle(train_data)
      
      train_data = list(self.accu_train_data.values())

      # save current network, then train to get a new network
      self.curr_nnet.save_checkpoint(folder=self.args.checkpoint_dir, 
                                     filename='temp_nnet.pth.tar')
      self.prev_nnet.load_checkpoint(folder=self.args.checkpoint_dir, 
                                     filename='temp_nnet.pth.tar')

      # declare MCTSs using prevoius and current neural nets
      prev_mcts = MCTS(self.prev_nnet, self.args)
      self.curr_nnet.train(train_data)
      curr_mcts = MCTS(self.curr_nnet, self.args)

      # pit against previous model
      log.info('Pit against previous module')
      arena = Arena(self.args, prev_mcts, curr_mcts, self.game)
      prev_wins, curr_wins, ties = arena.playGames(self.n_arena_games)
      sum_wins = prev_wins + curr_wins
      win_rate = (float(curr_wins) / float(sum_wins)) if sum_wins > 0 else 0
      log.info('Curr / Prev wins: %d / %d; Ties: %d'% (curr_wins, prev_wins, ties))
      log.info('Win rate %.2f'% win_rate)

      # accept or reject current new model
      if (sum_wins == 0) or (win_rate < self.args.accept_threshold):
        log.info('Reject new model')
        self.curr_nnet.load_checkpoint(folder=self.args.checkpoint_dir, 
                                       filename='temp_nnet.pth.tar')
      else:
        log.info('Accept new model')
        # self.curr_nnet.save_checkpoint(folder=self.args.checkpoint_dir,
        #                                filename=self.getFilenameIteration(i))
        self.curr_nnet.save_checkpoint(folder=self.args.checkpoint_dir,
                                       filename='best_nnet.pth.tar')

      log.info(f'Completed training iteration: {i}')

  def getFilenameIteration(self, iteration):
    return 'checkpoint_' + str(iteration) + '.pth.tar'

  def saveTrainData(self, iteration):
    folder = self.args.checkpoint_dir
    if not os.path.exists(folder):
      os.makedirs(folder)
    filename = os.path.join(folder, 
                            self.getFilenameIteration(iteration) + ".samples")
    with open(filename, "wb+") as f:
      Pickler(f).dump(self.accu_train_data)
    f.closed

  def loadTrainData(self):
    model = os.path.join(self.args.pretrained_model_file[0], 
                         self.args.pretrained_model_file[1])
    samples = model + ".samples"
    if not os.path.isfile(samples):
      log.warning(f'No training data: {samples}!')
      r = input("Continue training? [Y/n]")
      if r != "y":
        sys.exit()
    else:
      log.info("Starting loading training data")
      with open(samples, "rb") as f:
        self.accu_train_data = Unpickler(f).load()
      log.info('Completed loading training data')

  def loadTrainDataFromText(self):
    fp = open('./data/states.txt', 'r')
    data = []
    states = []
    for line in fp:
      if (line.startswith(' [') or line.startswith('[')):
        array_ends = True if line.endswith(']]\n') else False
        stripped = line.strip(' []\n')
        data.append(stripped)
        if (array_ends):
          states.append(np.loadtxt(data, dtype=int))
          data = []
    fp.close()

    fp = open('./data/pis.txt', 'r')
    pis = []
    for line in fp:
      stripped = line.strip(' pi:[]\n')
      pis.append(np.loadtxt([stripped], dtype=float))
    fp.close()

    fp = open('./data/vs.txt', 'r')
    vs = []
    for line in fp:
      stripped = line.strip(' v:\n')
      vs.append(float(stripped))
    fp.close()

    assert(len(states) == len(pis))
    assert(len(pis) == len(vs))
    samples = list(tuple(zip(states, pis, vs)))
    self.accu_train_data.append(samples)

  def learnFromData(self):
    train_data = []
    for e in self.accu_train_data:
      train_data.extend(e)
    # shuffle data and train model
    shuffle(train_data)
    self.curr_nnet.train(train_data)
