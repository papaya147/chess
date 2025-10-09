from state import board_to_tensor, move_mask, move_to_index, index_to_move
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess
from network import ChessNetV2 as ChessNet
from device import device
import os

class MCTSNode:
    def __init__(self, board, parent=None, prior=0.0, move_idx=None):
        self.board = board.copy()
        self.parent = parent
        self.prior = prior
        self.move_idx = move_idx
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.is_expanded = False

    def average_value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0.0
    
def expand_node(node, net, board, alpha=None, epsilon=None, policy_logits=None, value=None):
    if policy_logits is None or value is None:
        state = board_to_tensor(board).unsqueeze(0)
        policy_logits, value = net(state)
        policy_logits = policy_logits.squeeze(0)
        value = value.item()

    mask = torch.tensor(move_mask(board), dtype=torch.float32, device=device)
    masked_logits = policy_logits + (mask - 1) * 1e9
    prior = F.softmax(masked_logits, dim=0).cpu().detach().numpy()

    if alpha is not None and epsilon is not None:
        legal_idxs = np.where(mask.cpu().numpy() == 1)[0]
        if len(legal_idxs) > 0:
            noise = np.random.dirichlet([alpha] * len(legal_idxs))
            prior[legal_idxs] = (1 - epsilon) * prior[legal_idxs] + epsilon * noise

    for idx in np.where(mask.cpu().numpy() == 1)[0]: # add all children nodes (create child wherever legal move exists)
        move_uci = index_to_move[idx]
        child_board = board.copy()
        child_board.push_uci(move_uci)
        node.children[idx] = MCTSNode(child_board, parent=node, prior=prior[idx], move_idx=idx)

    node.is_expanded = True
    node.visits += 1
    node.value_sum += value if board.turn == chess.WHITE else -value

    return value if board.turn == chess.WHITE else -value

def batch_expand_nodes(nodes, net, boards, alpha=None, epsilon=None):
    states = torch.stack([board_to_tensor(board) for board in boards]).to(device)
    policy_logits, values = net(states)
    results = []
    for i, node in enumerate(nodes):
        board = boards[i]
        value = expand_node(
            node,
            net,
            board,
            alpha=alpha,
            epsilon=epsilon,
            policy_logits=policy_logits[i],
            value=values[i].item()
        )
        results.append(value)
    return results

def select_node(root, c_puct):
    node = root
    while node.is_expanded and node.children:
        best_score = -float('inf')
        best_child = None

        for child in node.children.values():
            q = child.average_value()
            u = c_puct * child.prior * (node.visits ** 0.5) / (1 + child.visits)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child
        
        if best_child is None:
            break
        node = best_child
    return node

def terminal_value(board):
    result = board.result()
    if result == '1-0':
        return 1.0 if board.turn == chess.BLACK else -1.0 # winner is opp of curr turn
    elif result == '0-1':
        return 1.0 if board.turn == chess.WHITE else -1.0
    return 0.0

def backpropogate(node, value):
    current = node
    while current:
        current.visits += 1
        current.value_sum += value if current.board.turn == chess.WHITE else -value
        value = -value # negate for opp in next iter
        current = current.parent

def visit_policy(root, temperature, device=torch.device('mps')):
    visits = np.zeros(len(move_to_index))
    for idx, child in root.children.items():
        visits[idx] = child.visits
    
    if temperature == 0:
        probs = np.zeros_like(visits)
        probs[np.argmax(visits)] = 1.0
    else:
        visits = visits ** (1 / temperature)
        probs = visits / visits.sum()
    
    return torch.tensor(probs, dtype=torch.float32, device=device)
    
def search(net, board, n_sims=400, batch_size=32, c_puct=1.5, temperature=1.5, alpha=0.2, epsilon=0.2):
    root = MCTSNode(board)

    expand_node(root, net, board, alpha, epsilon)

    for _ in range(0, n_sims - 1, batch_size):
        nodes = []
        boards = []

        for _ in range(min(batch_size, n_sims - 1 - len(nodes))):
            node = select_node(root, c_puct)
            if node.board.is_game_over():
                value = terminal_value(node.board)
                backpropogate(node, value)
            else:
                nodes.append(node)
                boards.append(node.board)
        
        if nodes:
            values = batch_expand_nodes(nodes, net, boards, alpha=alpha, epsilon=epsilon)
            for node, value in zip(nodes, values):
                backpropogate(node, value)

    probs = visit_policy(root, temperature)

    value = terminal_value(root.board) if root.board.is_game_over() else root.average_value()

    return probs, value

def selfplay(net, n_sims, game_save_path, c_puct=1.5, temperature=1.5, alpha=0.2, epsilon=0.2):
    board = chess.Board()
    boards, positions, policies, values = [], [], [], []

    while not board.is_game_over():
        probs, value = search(net, board, n_sims, c_puct=c_puct, temperature=temperature, alpha=alpha, epsilon=epsilon)
        if probs.sum() == 0:
            print('no valid moves')
            break
        
        move_idx = torch.multinomial(probs, 1).item()
        move_uci = index_to_move[move_idx]

        boards.append(board.copy())
        state = board_to_tensor(board)

        board.push_uci(move_uci)

        positions.append(state.cpu())
        policies.append(probs.cpu().numpy())
        values.append(value)
    
    # final reward
    result = board.result()
    reward = 1 if result == '1-0' else -1 if result == '0-1' else 0
    
    final_values = []
    for i in range(len(values)):
        if i % 2 == 0:  # white's turn
            final_values.append(reward)
        else:  # black's turn
            final_values.append(-reward)

    with open(game_save_path, 'w') as file:
        file.write(', '.join([str(move) for move in board.move_stack]))

    return boards, positions, policies, final_values

def selfplay_wrapper(net_path, num_sims, game_save_path, c_puct, temperature, alpha, epsilon):
    net = ChessNet(n_moves=len(move_to_index))
    net.load_state_dict(torch.load(net_path))
    net = net.to(device)
    net.eval()
    return selfplay(net, num_sims, game_save_path, c_puct, temperature, alpha, epsilon)