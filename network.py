import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import pickle
import os
import chess

from pettingzoo.classic.chess import chess_utils as ut


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def label_moves(df):
    df['player'] = [(8 if i % 2 == 0 else 9) for i in range(len(df))]
    return df


def move_map_black(move):
    TOTAL = 73
    source = move.from_square
    coord = ut.square_to_coord(source)
    panel = ut.get_move_plane(move)
    cur_action = (coord[0] * 8 + coord[1]) * TOTAL + panel
    return cur_action


def move_map_white(uci_move):
    TOTAL = 73
    move = chess.Move.from_uci(uci_move)
    source = move.from_square
    coord = ut.square_to_coord(source)
    panel = ut.get_move_plane(move)
    cur_action = (coord[0] * 8 + coord[1]) * TOTAL + panel
    return cur_action


def black_move(uci_move):
    move = chess.Move.from_uci(uci_move)
    mir = ut.mirror_move(move)
    return mir


class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        # common layers
        self.conv1 = nn.Conv2d(111, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*board_width*board_height,
                                 board_width*board_height*73)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*board_width*board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.board_width*self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        return x_act, x_val


class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height, model_file=None):
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty

        if torch.cuda.is_available():  # Windows
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():  # Mac OS
            self.device = torch.device("mps")
        else:  # CPU
            self.device = torch.device("cpu")

        # the policy value net module
        self.policy_value_net = Net(board_width, board_height).to(self.device)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)
        if model_file:
            net_params = torch.load(model_file, map_location=self.device, weights_only=True)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        state_batch = np.array(state_batch)
        state_batch = torch.tensor(state_batch, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = torch.exp(log_act_probs).cpu().numpy()

        return act_probs, value

    def policy_value_fn(self, env, state):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        available = []
        uci_moves = list(env.env.env.env.board.legal_moves)
        uci_moves = [move.uci() for move in uci_moves]
        if env.env.env.env.board.turn == True:
            for uci_move in uci_moves:
                available.append(move_map_white(uci_move))
        else:
            for uci_move in uci_moves:
                move = black_move(uci_move)
                available.append(move_map_black(move))

        current_state = torch.tensor(state.copy(), dtype=torch.float32)
        current_state = current_state.permute(2, 0, 1).unsqueeze(0).to(self.device)  # (8, 8, 111) -> (1, 111, 8, 8)

        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(current_state)
            act_probs = torch.exp(log_act_probs).cpu().numpy().flatten()
            masked_act_probs = np.zeros_like(act_probs)
            masked_act_probs[available] = act_probs[available]
            if masked_act_probs.sum() > 0:  # if have not available action
                masked_act_probs /= masked_act_probs.sum()
            else:
                masked_act_probs /= (masked_act_probs.sum()+1)

        return available, masked_act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        state_batch_np = np.array(state_batch)
        mcts_probs_np = np.array(mcts_probs)
        winner_batch_np = np.array(winner_batch)

        # numpy array to tensor
        state_batch = torch.tensor(state_batch_np, dtype=torch.float32, device=self.device)
        mcts_probs = torch.tensor(mcts_probs_np, dtype=torch.float32, device=self.device)
        winner_batch = torch.tensor(winner_batch_np, dtype=torch.float32, device=self.device)

        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer

        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss

        self.optimizer.zero_grad()
        set_learning_rate(self.optimizer, lr)

        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )
        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        # Ensure that the directory exists before saving the file
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        torch.save(net_params, model_file)
