import logging
import random
import torch
import numpy as np
import argparse
import gym
from model import SimpleModel
from trainer import train

logging.basicConfig(format='%(asctime)s -  %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
def main(args):
    env = gym.make(args.env)
    model = SimpleModel(env.action_space.n,4)
    optimizer= torch.optim.RMSprop(model.parameters(), lr=args.lr)
    train(model, optimizer, args, env)
    logger.info('******************** Train Finished ********************')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_set', action='store_true', default=False)
    parser.add_argument('--env', default="CartPole-v1")
    parser.add_argument('--epsilon', type= float, default= 0.3)
    parser.add_argument('--epsilon_decay_epoch', type=float, default=1)


    # training
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)

    main(args)