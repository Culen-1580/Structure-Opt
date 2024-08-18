import copy
import sys
import logging
from tqdm import tqdm
import numpy as np
from mcts import MCTS
from graph_game import GraphGame as Game
from dot_dict import DotDict

log = logging.getLogger(__name__)

class Arena():
  """
  pit 2 agents/models against each other.
  """
  def __init__(self, args: DotDict, prev_mcts: MCTS, curr_mcts: MCTS, game: Game):
    self.EPS = args.EPS
    self.prev_mcts = prev_mcts
    self.curr_mcts = curr_mcts
    self.prev_results = []
    self.curr_results = []
    self.game = game
    self.best_result = sys.maxsize

  def playGame(self, mcts: MCTS) -> int:
    """
    Play 1 episode/round of the game.
    """
    game = copy.copy(self.game)
    # results collected during current episode
    result = game.getGraphSize()
    while not game.isGameEnded():
      pi = mcts.getActionProb(game, mode=0)
      # check whether action probability vector is zero or not
      if (np.sum(pi) < self.EPS):
        break
      action = int(np.argmax(pi))
      game.getNextState(action)
      result = min(result, game.getGraphSize())
    return result

  def playGames(self, n_epsd_pair):
    """
    Play '2*n_epsd_pair' game episodes/rounds [0, 1, ..., 2*n_epsd_pair-1].
    Player1 plays episode [0, 2, ...], and Player2 plays episode [1, 3, ...]
    Compare results of episode pair (2*i, 2*i+1) to determine which player wins
    Returns:
      num_win1: number of episode pair won by player1
      num_win2: number of episode pair won by player2
      num_ties: number of episode pair with tie
      'num_win1' + 'num_win2' + 'num_tie' = 'n_epsd_pair'
    """
    num_win1 = 0
    num_win2 = 0
    num_tie = 0
    for _ in tqdm(range(n_epsd_pair), desc="Arena pitting"):
      result1 = self.playGame(self.prev_mcts)
      result2 = self.playGame(self.curr_mcts)
      print(f'result1: {result1}, result2: {result2}')
      self.prev_results.append(result1)
      self.curr_results.append(result2)
      self.best_result = min(self.best_result, min(result1, result2))
      if result1 < result2:
        num_win1 += 1
      elif result1 > result2:
        num_win2 += 1
      else:
        num_tie += 1
    print(f'prev_results: {self.prev_results}')
    print(f'curr_results: {self.curr_results}')
    print(f'best_result: {self.best_result}\n')
    return num_win1, num_win2, num_tie
