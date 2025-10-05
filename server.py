from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from network import ChessNet
from state import move_to_index, index_to_move, board_to_tensor, move_mask
import chess
from device import device
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

n_moves = len(move_to_index)
model_path = 'modelv1.pth'

model = ChessNet(n_moves)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        fen = data.get('fen')
        if not fen or not isinstance(fen, str):
            return jsonify({
                'error': 'no valid FEN provided'
            }), 400
        
        board = chess.Board(fen)
        state = board_to_tensor(board).unsqueeze(0)

        mask = move_mask(board)

        with torch.no_grad():
            policy, value = model(state)
            policy = policy.cpu()
            value = value.cpu().numpy().flatten()[0]

        masked_policy = policy + (mask - 1) * 1e9
        masked_policy = F.softmax(masked_policy)

        chosen_idx = np.random.choice(range(len(masked_policy[0])), p=masked_policy[0].numpy())
        chosen_move = index_to_move[chosen_idx]
        board.push_uci(chosen_move)

        return jsonify({
            'policy': policy.tolist(),
            'value': float(value),
            'chosen_move': chosen_move,
            'fen': board.fen(),
            'status': 'success'
        })
        
    except Exception as e:
        print(e)
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)