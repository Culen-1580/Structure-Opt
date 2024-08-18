import logging
import copy
import math
import numpy as np
import dot_dict as DotDict
import graph_game as Game
from neural_net_wrapper import NeuralNetWrapper as Net

log = logging.getLogger(__name__)

class MCTS():
  """
  Monte-Carlo Tree Search
  """
  def __init__(self, nnet: Net, args: DotDict):
    self.nnet = nnet
    self.args = args
    self.EPS = args.EPS
    self.cpuct = args.cpuct
    self.Q = {} # Q[s][a]: reward value obtained by searching from state-action pair '(s,a)'
    self.N = {} # N[s][a]: number of times that state-action pair '(s,a)' is visited
    self.P = {} # P[s][a]: prob of choosing action 'a' at state 's', policy from neural net output
    self.V = {} # V[s][a]: =='1' means choosing action 'a' from state 's' is valid, =='0' otherwise

  def getActionProb(self, game: Game, mode: int = 1, step: int = 0) -> np.array:
    """
    Performs 'n_MCTS' MCTS simulations starting from current game state.
    Returns:
      probs: vector with action 'a' probability = N[s]
    """
    for _ in range(self.args.n_MCTS):
      # deep copy of current graph
      game_dup = copy.copy(game)
      # reset action count, since training mode has an infinite number of steps
      if (mode == 1):
        game_dup.act_count = 0
      self.search(game_dup)

    s = game.getOrderByEdges()
    counts = self.N[s]

    ### for a in range(game.getActionSize()):
    ###   if (s, a) in self.Nsa:
    ###     print(s, '  ', a, '  ', self.Ns[s], '  ', self.Nsa[(s, a)])
    ###     np.set_printoptions(precision=3)

    probs = np.zeros(len(counts))
    sum_counts = np.sum(counts)
    if (sum_counts < self.EPS):
      return probs

    '''
    if (mode == 0): # inference
      # TODO: only focus on 'bestA' for inference ?
      bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
      bestA = np.random.choice(bestAs)
      probs[bestA] = 1
    else: # training
      probs = np.array(counts)
      probs /= sum_counts
    '''
    probs = np.array(counts)
    probs /= sum_counts

    valids = game.getValidActions()
    for i in range(len(valids)):
      if ((valids[i] < self.EPS) and (probs[i] > self.EPS)):
        print('\ngetActionProb')
        print(f'valids: {valids}')
        print(f'probs: {probs}')
        assert(False)
    
    # print info
    if ((mode == 1) and ((step % self.args.sampling_dis) == 1)):
      print(f'  order: {s}')
      if (s in self.P):
        np.set_printoptions(precision=2)
        print(f'  orig prob vector: {self.P[s]}')
      else:
        print('  no orig prob vector!')
      np.set_printoptions(precision=2)
      print(f'  mcts prob vector: {probs}')
    return probs

  def search(self, game: Game):
    reward = game.getReward()
    if game.isGameEnded():
      return reward
    
    act_size = game.getActionSize()
    s = game.getOrderByEdges()
    # state 's' is encountered for the first time
    if s not in self.P:
      state = game.getStateInNpArray()
      self.V[s], self.Q[s] = game.getValidActionsAndRewards()
      self.P[s] = (self.nnet.predict(state) * self.V[s]) # mask invalid actions
      sum_prob = np.sum(self.P[s])
      if (sum_prob < self.EPS):
        # All valid actions have probability 0 to be selected.
        # potential reasons:
        # 1). Neural net is insufficient; 2). Overfitting.
        # Many this message means issues with neural net and/or training process.  
        log.error("All valid actions have probability 0 to be selected.")
        print("All valid actions have probability 0 to be selected.")
        # set equal probability for all valid actions
        # at least action 0 will be valid
        self.P[s] = np.array(self.V[s])
        sum_prob = np.sum(self.P[s])
      assert(sum_prob)
      self.P[s] /= sum_prob  # normalize
      self.N[s] = np.zeros(act_size)
      return reward

    moves = self.V[s]
    sum_moves = np.sum(moves)
    assert(sum_moves > self.EPS) # at least can take action 0
    sum_rewards = np.sum(self.Q[s])
    assert(sum_rewards > self.EPS) # at least action 0 has positive reward

    '''
    best_score = -float('inf')
    best_act = -1
    # choose the action with highest upper confidence bound
    for a in range(act_size):
      if moves[a] > self.EPS:
        u = (self.Q[s][a] / sum_rewards) + (self.cpuct * self.P[s][a] / (1 + self.N[s][a]))
        if u > best_score:
          best_score = u
          best_act = a
    '''
    scores = (self.Q[s] / sum_rewards) + (self.cpuct * self.P[s] / (1 + self.N[s]))
    sum_scores = np.sum(scores)
    assert(sum_scores > self.EPS)
    scores = scores / sum_scores
    best_act = np.random.choice(len(scores), p=scores)

    game.getNextState(best_act)
    order = game.getOrderByEdges()
    reward = max(reward, self.search(game))

    a = best_act
    assert(moves[a] > self.EPS)
    self.Q[s][a] = (self.N[s][a] * self.Q[s][a] + reward) / (1 + self.N[s][a])
    self.N[s][a] += 1
    return reward
