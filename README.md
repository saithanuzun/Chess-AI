# Chess-AI

This project is a Python‑based chess engine powered by deep learning. It uses a convolutional neural network trained on 20,000 human games and approximately 40,000 chess positions to predict the best move in any given position. The engine receives input as a FEN string, which is converted into a 13×8×8 PyTorch tensor. It then classifies over the full 64×63 UCI‑legal move space to produce its predictions. The model is treated as a classification problem, cross‑entropy loss and the Adam optimizer are used to train. Training ran for roughly 12 hours and produced strong accuracy. Although the network occasionally outputs illegal moves, future improvements such as masking legal moves could further enhance its reliability.

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
<script src="https://cdn.jsdelivr.net/npm/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/1.0.0/chess.min.js"></script>

<div class="chess-ai-container">
  <h2>Chess AI Predicted Moves</h2>

  <div id="boards-container"></div>
</div>

<style>
.chess-ai-container {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  max-width: 900px;
  margin: 20px auto;
  padding: 10px;
}

.chess-ai-container h2 {
  text-align: center;
  color: #2c3e50;
}

.board-card {
  display: flex;
  align-items: center;
  gap: 20px;
  margin-bottom: 20px;
}

.board-card .prediction {
  font-size: 1rem;
  color: #27ae60;
  font-weight: bold;
}
</style>

<script>
const data = [
  { fen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', move: 'e2e4' },
  { fen: 'rnbqkbnr/ppp1pppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2', move: 'd2d4' },
  { fen: 'r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3', move: 'a7a6' },
  { fen: 'r2q1rk1/ppp2ppp/2n2n2/2b1p3/4P3/2N1BN2/PPP2PPP/R2Q1RK1 w - - 6 10', move: 'e3c5' },
  { fen: 'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2', move: 'g1f3' },
  { fen: 'rnbqkb1r/pppp1ppp/5n2/8/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 2 3', move: 'b1c3' },
  { fen: 'r1bq1rk1/ppp2ppp/2n1pn2/3p4/3P4/2P1PN2/P1Q2PPP/R1B1KB1R w KQ - 0 8', move: 'f1d3' },
  { fen: '8/8/8/4k3/8/4P3/4K3/8 w - - 0 30', move: 'e2f3' },
  { fen: '8/8/8/4R3/4k3/8/4K3/8 w - - 0 50', move: 'e5e3' },
  { fen: 'r1bq1rk1/pp1n1ppp/2p1pn2/3p4/3P1B2/2N1PN2/PPQ2PPP/R3KB1R b KQ - 2 10', move: 'f8e8' },
];

const container = document.getElementById('boards-container');

data.forEach((item, index) => {
  const card = document.createElement('div');
  card.className = 'board-card';

  const boardDiv = document.createElement('div');
  boardDiv.id = 'board-' + index;
  boardDiv.style.width = '200px';

  const prediction = document.createElement('div');
  prediction.className = 'prediction';
  prediction.innerHTML = `Predicted move: ${item.move.toUpperCase()}`;

  card.appendChild(boardDiv);
  card.appendChild(prediction);
  container.appendChild(card);

  // Initialize Chessboard
  Chessboard(boardDiv.id, {
    position: item.fen,
    draggable: false
  });
});
</script>
