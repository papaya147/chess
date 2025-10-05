
# Chess Bot
Welcome to Chess Bot, a web-based chess game where you can challenge an AI opponent powered by reinforcement learning (RL). This project combines a Monte Carlo Tree Search (MCTS) algorithm, a neural network, and replay buffers to create a smart and engaging chess-playing bot. The bot runs on a Flask backend and features a simple HTML frontend for a browser-based experience.

## About the Bot
The Chess Bot uses advanced reinforcement learning techniques to play chess. Here's what powers it:
-   **Neural Network**: A PyTorch-based convolutional neural network (CNN) evaluates board positions, predicting move probabilities (policy) and win likelihood (value).
-   **Monte Carlo Tree Search (MCTS)**: Guides the bot's decision-making by simulating thousands of possible game outcomes, balancing exploration and exploitation.
-   **Replay Buffer**: Stores game experiences (states, actions, rewards) for training, allowing the bot to learn and improve over time.

Training is very resource and time-intensive, but the notebook [`train.ipynb`](https://github.com/papaya147/chess-bot/blob/main/train.ipynb) can be modified to train your own bot.

## Playing Against the Bot
Playing the Chess Bot is easy and requires only a web browser. Follow these steps to get started:
### Prerequisites

-   A modern web browser (e.g., Chrome, Firefox, Safari)
-   Conda (for Python environment management)

### Setup
1. Clone the Repository:
```bash
git clone https://github.com/papaya147/chess-bot.git
cd chess-bot
```
2. Install Dependencies:
```bash
conda  create  --prefix  ./env  python=3.11
conda  activate  ./env
pip install -r requirements.txt
```
3. Run the backend (uses port 8080)
```bash
python server.py
```
4. Open up the HTML page
For macOS:
```bash
open index.html
```
For Windows:
```bash
start index.html
```
For Linux:
```bash
xdg-open index.html
```

### Playing the Game
-   **Start**: You play as White by default (bot as Black).
-   **Make Moves**: Click or drag pieces to make legal moves, validated by the frontend and backend.
-   **Bot Response**: The bot uses MCTS and its neural network to respond, typically within seconds.
-   **Game End**: The game detects checkmate, stalemate, or draws automatically.
-   **Extras**: Refresh the page to start over.
