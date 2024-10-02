import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from pettingzoo.classic.chess import chess_utils


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        # common layers
        self.conv1 = nn.Conv2d(111, 32, kernel_size=3, padding=1)  # TODO board.shape (8 * 8 * 111) -> (1 * 111 * 8 * 8)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*board_width*board_height,  # TODO act_fc1이 이렇게 넣어주면 안되는거 일 수도 있음. 일단 나중에 다시
                                 board_width*board_height*73) # TODO action space가 8 * 8 * 73이라서
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
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        state_batch = np.array(state_batch)
        state_batch = torch.tensor(state_batch, dtype=torch.float32, device=self.device)
        log_act_probs, value = self.policy_value_net(state_batch)
        act_probs = torch.exp(log_act_probs).cpu().numpy()
        # TODO  numpy나 cpu로 바꿔서 넣어줘야할수도 있음.
        # value = value.cpu().numpy()
        return act_probs, value

    def policy_value_fn(self, env, agent):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        uci_moves = list(env.env.env.env.board.legal_moves)
        uci_moves = [move.uci() for move in uci_moves]
        for uci_move in uci_moves:
            legal_positions = chess_utils.make_move_mapping(uci_move)

        legal_positions = list(legal_positions.values())
        current_state = env.observe(agent)['observation']
        current_state = torch.tensor(current_state, dtype=torch.float32)
        current_state = current_state.permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(current_state)
            act_probs = torch.exp(log_act_probs).cpu().numpy().flatten()

        # TODO legal_position 에 있는 정보들을 숫자로 바꿔서 거기에 맞는 act_probs랑 맞춰줘야하나?
        print(legal_positions)
        act_probs = zip(legal_positions, act_probs[legal_positions]) # TODO 오목에서는 그냥 가능한 action에다가 확률값 붙여주긴했는데 action_mask를 붙여주면 되나? 어떻게 하지
        return act_probs, value # TODO 여기서 또 궁금한점 지금 이대로 하면  가능한 action들이랑, 그 action 들의 확률값을 zip해서 넘겨주는데 act_probs[legal_positions] 이 확률값 다 더했을때 1이 되어야하는거 아닌가?


    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        state_batch_np = np.array(state_batch)
        mcts_probs_np = np.array(mcts_probs)
        winner_batch_np = np.array(winner_batch)

        # numpy array to tensor
        state_batch = torch.tensor(state_batch_np, dtype=torch.float32, device=self.device)
        mcts_probs = torch.tensor(mcts_probs_np, dtype=torch.float32, device=self.device)
        winner_batch = torch.tensor(winner_batch_np, dtype=torch.float32, device=self.device)

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss

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
        torch.save(net_params, model_file)