# 이 버전은 playout에서 env.reset을 매번 해줘야해서 함수를 하나로 만들어버리기로 함. 그 직전의 버전

import torch
import argparse
import numpy as np
import datetime
import random
import os

from pettingzoo.classic import chess_v6
from mcts_AlphaZero import MCTSPlayer
from network import PolicyValueNet
from collections import defaultdict, deque



parser = argparse.ArgumentParser()

""" MCTS parameter """
parser.add_argument("--buffer_size", type=int, default=10000)
parser.add_argument("--c_puct", type=int, default=5)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr_multiplier", type=float, default=1.0)
parser.add_argument("--n_playout", type=int, default=2)
parser.add_argument("--self_play_sizes", type=int, default=100)
parser.add_argument("--training_iterations", type=int, default=1500) # fiar에서는 100번했음
parser.add_argument("--temp", type=float, default=1.0)

""" Policy update parameter """
parser.add_argument("--batch_size", type=int, default=256)   # fiar에서는 64번했음
parser.add_argument("--data_buffers", type=int, default=deque(maxlen=10000))
parser.add_argument("--learn_rate", type=float, default=2e-3)
parser.add_argument("--lr_mul", type=float, default=1.0)
parser.add_argument("--kl_targ", type=float, default=0.02)  # 이게 train network에서 self.goal = 0.02이랑 같은거 인지 봐야할듯

""" Policy evaluate parameter """
parser.add_argument("--win_ratio", type=float, default=0.0)
parser.add_argument("--init_model", type=str, default=None)

args = parser.parse_args()

# make all args to variables
n_playout = args.n_playout
buffer_size = args.buffer_size
c_puct = args.c_puct
epochs = args.epochs
self_play_sizes = args.self_play_sizes
training_iterations = args.training_iterations
temp = args.temp
batch_size = args.batch_size
learn_rate = args.learn_rate
lr_mul = args.lr_mul
lr_multiplier = args.lr_multiplier
kl_targ = args.kl_targ
win_ratio = args.win_ratio
init_model = args.init_model


def get_equi_data(play_data):
    """augment the data set by rotation and flipping
    play_data: [(state, mcts_prob, winner_z), ..., ...]
    """
    extend_data = []
    board_height = 8
    board_width = 8
    for state, mcts_prob, winner in play_data:
        for i in [1, 2, 3, 4]:
            # rotate counterclockwise
            equi_state = np.array([np.rot90(s, i) for s in state])
            equi_mcts_prob = np.rot90(np.flipud(
            mcts_prob.reshape(board_height, board_width)), i)
            extend_data.append((equi_state,
                                np.flipud(equi_mcts_prob).flatten(),
                                winner))
            # flip horizontally
            equi_state = np.array([np.fliplr(s) for s in equi_state])
            equi_mcts_prob = np.fliplr(equi_mcts_prob)
            extend_data.append((equi_state,
                                np.flipud(equi_mcts_prob).flatten(),
                                winner))
    return extend_data


def collect_selfplay_data(env, mcts_player, n_games=100):
    """collect self-play data for training"""
    for i in range(n_games):
        winner, play_data = self_play(env, mcts_player, temp)
        play_data = list(play_data)[:]
        episode_len = len(play_data)
        print("in self-play episode 길이:", episode_len)
        # augment the data
        play_data = get_equi_data(play_data)
        data_buffer.extend(play_data)


def self_play(env, mcts_player, temp):
    env.reset()
    player_0 = 0
    player_1 = 1 - player_0
    states, mcts_probs, current_player = [], [], []
    move_list = []

    while True:
        observation, reward, termination, truncation, info = env.last()
        current_state = torch.tensor(observation['observation'].copy(), dtype=torch.float32)
        move, move_probs = mcts_player.get_action(env, current_state, move_list, temp, return_prob=1)

        # env.reset()

        move_list.append(move)
        observation_ = torch.permute(current_state, (2, 0, 1))  # (8, 8, 111) -> (111, 8, 8)
        states.append(observation_)
        mcts_probs.append(move_probs)
        current_player.append(player_0)

        # for i in range(len(move_list)):  # move_list의 길이만큼 반복
            # move = move_list[i]  # 인덱스를 사용하여 move_list에서 move를 가져옴
            # env.step(move)

        player_0 = 1 - player_0
        player_1 = 1 - player_0

        if termination or truncation:
            if env.rewards == 0:  # TODO 수정해야함
                print('self_play_draw')
            mcts_player.reset_player()  # reset MCTS root node
            winners_z = np.zeros(len(current_player))

            if env.rewards != 0:  # non draw
                if env.rewards == -1:
                    env.rewards = 0
                # if winner is current player, winner_z = 1
                winners_z[np.array(current_player) == 1 - env.rewards] = 1.0
                winners_z[np.array(current_player) != 1 - env.rewards] = -1.0

            return env.rewards, zip(states, mcts_probs, winners_z)


def policy_update(lr_mul, policy_value_net, data_buffer):
    """update the policy-value net"""
    lr_multiplier = lr_mul

    mini_batch = random.sample(data_buffer, batch_size)
    state_batch = [data[0] for data in mini_batch]
    mcts_probs_batch = [data[1] for data in mini_batch]
    winner_batch = [data[2] for data in mini_batch]
    old_probs, old_v = policy_value_net.policy_value(state_batch)

    for i in range(epochs):
        loss, entropy = policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                learn_rate*lr_multiplier)
        new_probs, new_v = policy_value_net.policy_value(state_batch)
        kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                axis=1)
        )
        if kl > kl_targ * 4:  # early stopping if D_KL diverges badly
            break
    # adaptively adjust the learning rate
    if kl > kl_targ * 2 and lr_multiplier > 0.1:
        lr_multiplier /= 1.5
    elif kl < kl_targ / 2 and lr_multiplier < 10:
        lr_multiplier *= 1.5

    # explained_var_old = (1 -
    #                      np.var(np.array(winner_batch) - old_v.flatten()) /
    #                      np.var(np.array(winner_batch)))
    # explained_var_new = (1 -
    #                      np.var(np.array(winner_batch) - new_v.flatten()) /
    #                      np.var(np.array(winner_batch)))
    print(("kl:{:.5f},"
           "lr_multiplier:{:.3f},"
           "loss:{},"
           "entropy:{},"
           # "explained_var_old:{:.3f},"
           # "explained_var_new:{:.3f}"
           ).format(kl,
                    lr_multiplier,
                    loss,
                    entropy))
                    # explained_var_old,
                    # explained_var_new
    return loss, entropy



def policy_evaluate(env, current_mcts_player, old_mcts_player, n_games=30):  # total 30 games
    """
    Evaluate the trained policy by playing against the pure MCTS player
    Note: this is only for monitoring the progress of training
    """
    training_mcts_player = current_mcts_player  # training Agent
    opponent_mcts_player = old_mcts_player
    # leaf_mcts_player = MCTS_leaf(policy_value_fn, c_puct=c_puct, n_playout=n_playout)
    # random_action_player = RandomAction()
    win_cnt = defaultdict(int)

    for j in range(n_games):
        # reset for each game
        winner = start_play(env, training_mcts_player, opponent_mcts_player)
        if winner == -0.5:
            winner = 0
        win_cnt[winner] += 1
        print("{} / 30 ".format(j + 1))

    win_ratio = 1.0 * win_cnt[1] / n_games
    print("---------- win: {}, lose: {}, tie:{} ----------".format(win_cnt[1], win_cnt[0], win_cnt[-1]))
    return win_ratio, training_mcts_player



if __name__ == '__main__':
    env = chess_v6.env()
    env.reset(seed=42)

    if torch.cuda.is_available():  # Windows
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # Mac OS
        device = torch.device("mps")
    else:  # CPU
        device = torch.device("cpu")

    # TODO 만약에 dataset을 써야한다면 바로 밑의  model_file=init_model 대신에 human data 파일 경로를 넣어주면 될 것
    if init_model:
        policy_value_net = PolicyValueNet(env.observe('player_0')['observation'].shape[0],
                                          env.observe('player_0')['observation'].shape[1],
                                          model_file=init_model)
    else:
        policy_value_net = PolicyValueNet(env.observe('player_0')['observation'].shape[0],
                                          env.observe('player_0')['observation'].shape[1])

    curr_mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct, n_playout, is_selfplay=1)
    data_buffers = deque(maxlen=10000) # TODO len 수정해야함
    try:
        for i in range(training_iterations):
            """collect self-play data each iteration 1500 games"""
            data_buffer = collect_selfplay_data(env, curr_mcts_player)
            data_buffers.append(data_buffer)

            """Policy update with data buffer"""
            loss, entropy, lr_multiplier, policy_value_net = policy_update(lr_mul=lr_multiplier,
                                                                           policy_value_net=policy_value_net,
                                                                           data_buffers=data_buffer)

            policy_evaluate(env, curr_mcts_player, curr_mcts_player)









    except KeyboardInterrupt:
        print('\n\rquit')