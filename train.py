import torch
import argparse
import numpy as np
import datetime
import random
import os
import time
import wandb


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
parser.add_argument("--n_playout", type=int, default=50)
parser.add_argument("--self_play_sizes", type=int, default=1)  # TODO 50으로 설정할거지만 일단 뒤에까지 가는데 오래걸려서 1로 잠깐 씀
parser.add_argument("--training_iterations", type=int, default=100)  # fiar에서는 100번했음
parser.add_argument("--temp", type=float, default=1.0)

""" Policy update parameter """
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--learn_rate", type=float, default=1e-3)
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
    channels = 73
    for state, mcts_prob, winner in play_data:
        for i in [1, 2, 3, 4]:
            # rotate counterclockwise
            equi_state = np.array([np.rot90(s, i) for s in state])
            equi_mcts_prob = np.rot90(np.flipud(
                mcts_prob.reshape(channels, board_height, board_width)), i)
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


def collect_selfplay_data(env, mcts_player, game_iter):
    """collect self-play data for training"""
    data_buffer = deque(maxlen= 400 * 50 * 8)  # 400 (max len) * 50 (selfplay games) * 8 (equi)
    win_cnt = defaultdict(int)

    for self_play_i in range(self_play_sizes):
        rewards, play_data = self_play(env, mcts_player, temp, game_iter, self_play_i)
        play_data = list(play_data)[:]
        episode_len = len(play_data)
        print("in self-play episode 길이:", episode_len)

        # augment the data
        play_data = get_equi_data(play_data)
        data_buffer.extend(play_data)

        win_cnt[rewards] += 1

    print("\n ---------- Self-Play win: {}, lose: {}, tie:{} ----------".format(win_cnt[1], win_cnt[-1], win_cnt[0]))
    print(len(data_buffer))

    win_ratio = 1.0 * win_cnt[1] / self_play_sizes
    print("Win rate : ", round(win_ratio * 100, 3), "%")
    wandb.log({"Win_Rate/self_play": round(win_ratio * 100, 3)})

    return data_buffer


def self_play(env, mcts_player, temp, game_iter=0, self_play_i=0):
    env.reset()
    start_time = time.time()

    player_0 = 0
    player_1 = 1 - player_0
    states, mcts_probs, current_player = [], [], []
    move_list = []

    while True:
        observation, reward, termination, truncation, info = env.last()
        current_state = torch.tensor(observation['observation'].copy(), dtype=torch.float32)
        move, move_probs = mcts_player.get_action(env, current_state, move_list, temp, return_prob=1)

        move_list.append(move)
        observation_ = torch.permute(current_state, (2, 0, 1))  # (8, 8, 111) -> (111, 8, 8)
        states.append(observation_)
        mcts_probs.append(move_probs)
        current_player.append(player_0)

        env.step(move)

        player_0 = 1 - player_0
        player_1 = 1 - player_0
        observation, reward, termination, truncation, info = env.last()

        print(len(states)) # TODO 여기에서 len 찍고 있음

        if termination or truncation:
            # recode time
            iteration_time = time.time() - start_time
            formatted_time = time.strftime("%H:%M:%S", time.gmtime(iteration_time))
            print(formatted_time)
            wandb.log({"iteration_time": iteration_time})

            if reward == 0:
                print('self_play_draw')

            mcts_player.reset_player()  # reset MCTS root node
            print("game: {}, self_play:{}, episode_len:{}".format(
                game_iter + 1, self_play_i + 1, len(current_player)))
            winners_z = np.zeros(len(current_player))

            if reward != 0:  # non draw
                if reward == -1:
                    reward = 0
                # if winner is current player, winner_z = 1
                winners_z[np.array(current_player) == 1 - reward] = 1.0
                winners_z[np.array(current_player) != 1 - reward] = -1.0
                if reward == 0:
                    reward = -1
            return reward, zip(states, mcts_probs, winners_z)


def policy_update(lr_mul, policy_value_net, data_buffers=None):
    """update the policy-value net"""
    kl, loss, entropy = 0, 0, 0
    lr_multiplier = lr_mul
    update_data_buffer = [data for buffer in data_buffers for data in buffer]  # queue에 2번 집어넣어서 빼야힘

    mini_batch = random.sample(update_data_buffer, batch_size)
    state_batch = [data[0] for data in mini_batch]
    mcts_probs_batch = [data[1] for data in mini_batch]
    winner_batch = [data[2] for data in mini_batch]
    old_probs, old_v = policy_value_net.policy_value(state_batch)

    for i in range(epochs):
        loss, entropy = policy_value_net.train_step(state_batch,
                                                    mcts_probs_batch,
                                                    winner_batch,
                                                    learn_rate * lr_multiplier)
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

    print(("kl:{:.5f},"
           "lr_multiplier:{:.3f},"
           "loss:{},"
           "entropy:{}"
           ).format(kl, lr_multiplier, loss, entropy))

    return loss, entropy, lr_multiplier, policy_value_net


def policy_evaluate(env, current_mcts_player, old_mcts_player, n_games=30):  # total 30 games
    """
    Evaluate the trained policy by playing against the pure MCTS player
    Note: this is only for monitoring the progress of training
    """
    training_mcts_player = current_mcts_player  # training Agent
    opponent_mcts_player = old_mcts_player
    win_cnt = defaultdict(int)

    for j in range(n_games):
        winner = start_play(env, training_mcts_player, opponent_mcts_player)
        if winner == -1:
            winner = 0
        win_cnt[winner] += 1
        print("{} / 30 ".format(j + 1))

    win_ratio = 1.0 * win_cnt[1] / n_games
    print("---------- win: {}, lose: {}, tie:{} ----------".format(win_cnt[1], win_cnt[0], win_cnt[-1]))
    return win_ratio, training_mcts_player


def start_play(env, player1, player2):
    """start a game between two players"""
    env.reset()
    players = [0, 1]
    p1, p2 = players
    player1.set_player_ind(p1)
    player2.set_player_ind(p2)
    players = {p1: player1, p2: player2}
    current_player = 0
    player_in_turn = players[current_player]
    move_lists = []

    while True:
        observation, reward, termination, truncation, info = env.last()
        current_state = torch.tensor(observation['observation'].copy(), dtype=torch.float32)

        # synchronize the MCTS tree with the current state of the game
        move = player_in_turn.get_action(env, current_state, move_lists, temp=1e-3, return_prob=0)
        move_lists.append(move)
        print(len(move_lists))

        env.step(move)
        observation, reward, termination, truncation, info = env.last()

        if not termination or truncation:
            current_player = 1 - current_player
            player_in_turn = players[current_player]

        else:
            env.reset()
            return reward


if __name__ == '__main__':
    # wandb intialize
    wandb.init(mode="online",
               entity="hails",
               project="gym_chess",
               name="CHESS" + "-MCTS" + str(n_playout) + "-Date" + str(datetime.datetime.now()),
               config=args.__dict__
               )

    env = chess_v6.env()
    env.reset(seed=42)

    if torch.cuda.is_available():  # Windows
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # Mac OS
        device = torch.device("mps")
    else:  # CPU
        device = torch.device("cpu")

    if init_model:
        policy_value_net = PolicyValueNet(env.observe('player_0')['observation'].shape[0],
                                          env.observe('player_0')['observation'].shape[1],
                                          model_file=init_model)
    else:
        policy_value_net = PolicyValueNet(env.observe('player_0')['observation'].shape[0],
                                          env.observe('player_0')['observation'].shape[1])

    curr_mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct, n_playout, is_selfplay=1)
    data_buffer_training_iters = deque(maxlen=20)
    best_old_model, eval_model_file = None, None

    try:
        for i in range(training_iterations):
            """collect self-play data each iteration 1500 games"""
            data_buffer_each = collect_selfplay_data(env, curr_mcts_player, i)
            data_buffer_training_iters.append(data_buffer_each)

            """Policy update with data buffer"""
            loss, entropy, lr_multiplier, policy_value_net = policy_update(lr_mul=lr_multiplier,
                                                                           policy_value_net=policy_value_net,
                                                                           data_buffers=data_buffer_training_iters)
            if i == 0:
                policy_evaluate(env, curr_mcts_player, curr_mcts_player)

                model_file = f"Training/nmcts{n_playout}/train_{i + 1:03d}.pth"
                eval_model_file = f"Eval/nmcts{n_playout}/train_{i + 1:03d}.pth"

                policy_value_net.save_model(model_file)
                policy_value_net.save_model(eval_model_file)

            else:
                existing_files = [int(file.split('_')[-1].split('.')[0])
                                  for file in os.listdir(f"Training/nmcts{n_playout}")
                                  if file.startswith('train_')]
                old_i = max(existing_files)
                best_old_model = f"Training/nmcts{n_playout}/train_{old_i:03d}.pth"

            policy_value_net_old = PolicyValueNet(env.observe('player_0')['observation'].shape[0],
                                                  env.observe('player_0')['observation'].shape[1],
                                                  best_old_model)

            old_mcts_player = MCTSPlayer(policy_value_net_old.policy_value_fn, c_puct, n_playout,
                                         is_selfplay=0)

            if (i + 1) % 10 == 0:
                eval_model_file = f"Eval/nmcts{n_playout}/train_{i + 1:03d}.pth"

            policy_value_net.save_model(eval_model_file)
            print("Win rate : ", round(win_ratio * 100, 3), "%")

            if win_ratio > 0.5:
                old_mcts_player = curr_mcts_player
                model_file = f"Training/nmcts{n_playout}/train_{i + 1:03d}.pth"
                policy_value_net.save_model(model_file)
                print(" ---------- New best policy!!! ---------- ")

            else:
                # if worse it just reject and does not go back to the old policy
                print(" ---------- Low win-rate ---------- ")

    except KeyboardInterrupt:
        print('\n\rquit')