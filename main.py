import logging
import coloredlogs
import torch

from dot_dict import DotDict
from train import Train
from graph_game import GraphGame as Game
from neural_net_wrapper import NeuralNetWrapper as Net

log = logging.getLogger(__name__)
# can be changed to 'DEBUG' for more info
coloredlogs.install(level='INFO')
args = DotDict({
  # epsilon
  'EPS': 1e-8,
  # number of training iterations
  'n_iter': 50,
  # number of episodes/rounds of game in each iteration
  'n_epsds': 5,
  # number of steps for each episode/round in training
  'n_steps': 1000, #100000,
  # accept new neural net if winning rate > accept_threshold in Arena play
  'accept_threshold': 0.51, 
  # maximum number of training data samples to keep in each iteration
  # no need for this game, since # of samples obtained in each episode <= a fixed number
  'max_samples_per_iter': 300000,
  # number of MCTS simulations from each game state
  'n_MCTS': 200,
  # number of games/episodes in Arena play
  'n_arena_games': 20,
  # for MCTS
  'cpuct': 1.0,
  # folder to store generated models and training data during iterations
  'checkpoint_dir': './temp/',
  # control to load pretrained model or not
  'load_model': False,
  # control to load training data
  'load_train_data': False,
  # folder storing the pretrained models and training data
  'pretrained_model_file': ('./pretrained_models', 'best.pth.tar'),
  # maximum number of most recent iterations to keep for training data
  'max_iters_for_data': 10,
  # max number of actions for each round/episode to end in arena pitting
  'n_max_act': 100,
  # learning rate
  'lr': 0.01,
  # probability for dropout layer
  'dropout': 0.3,
  # at the end of each iteration, train the net 'epochs' times
  'epochs': 100,
  # batch size
  'batch_size': 10,
  # number of layers of graph neural net
  'n_layer': 20,
  # number of features for each graph node
  'n_feature': 1024,
  # max number of nodes allowed in graph, must be > # of nodes of initial graph
  'n_max_node': 80, #100,
  # use GPU to train
  'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
  # sampling distance for printing info
  'sampling_dis': 200
})

def main():
  # load game with initialization
  log.info('Game: %s', Game.__name__)
  game = Game(args, vars, expr)

  # load neural network/model
  log.info('Neural net: %s', Net.__name__)
  nnet = Net(args, game)
  if args.load_model:
    log.info('Load pretrained model "%s/%s"...', 
             args.pretrained_model_file[0], 
             args.pretrained_model_file[1])
    nnet.load_checkpoint(args.pretrained_model_file[0], 
                         args.pretrained_model_file[1])
  else:
    log.warning('No pretrained model loaded')

  log.info('Train')
  train = Train(args, game, nnet)

  # load pre-collected training data, if any
  if args.load_train_data:
    log.info('Load training data')
    # train.loadTrainData()
    train.loadTrainDataFromText()
    train.learnFromData()
  else:
    log.info('No training data loaded')

  # start training
  log.info('Starting training')
  train.learn()
  log.info('Completed training')

if __name__ == "__main__":
  main()
