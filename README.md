# Chess-AI

This project is a Python chess engine powered by deep learning. It uses a convolutional neural network trained on 20,000 human games and approximately 40,000 chess positions to predict the best move in any given position. The engine receives input as a FEN string, which is converted into a 13×8×8 PyTorch tensor. It then classifies over the full 64×63 UCI‑legal move space to produce its predictions. The model is treated as a classification problem, cross‑entropy loss and the Adam optimizer are used to train. Training ran for roughly 12 hours and produced strong accuracy. Although the network occasionally outputs illegal moves, future improvements such as masking legal moves could further enhance its reliability.

<!-- Chess AI Predictions - Testing Snippet -->
<div class="chess-ai-container">
  <h2>Chess AI Predicted Moves (Testing)</h2>
  <div class="chess-ai-grid">
    <div class="chess-card">
      <div class="fen">rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1</div>
      <div class="prediction">Predicted move: <strong>e2e4</strong></div>
    </div>
    <div class="chess-card">
      <div class="fen">rnbqkbnr/ppp1pppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2</div>
      <div class="prediction">Predicted move: <strong>d2d4</strong></div>
    </div>
    <div class="chess-card">
      <div class="fen">r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3</div>
      <div class="prediction">Predicted move: <strong>a7a6</strong></div>
    </div>
    <div class="chess-card">
      <div class="fen">r2q1rk1/ppp2ppp/2n2n2/2b1p3/4P3/2N1BN2/PPP2PPP/R2Q1RK1 w - - 6 10</div>
      <div class="prediction">Predicted move: <strong>e3c5</strong></div>
    </div>
    <div class="chess-card">
      <div class="fen">rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2</div>
      <div class="prediction">Predicted move: <strong>g1f3</strong></div>
    </div>
    <div class="chess-card">
      <div class="fen">rnbqkb1r/pppp1ppp/5n2/8/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 2 3</div>
      <div class="prediction">Predicted move: <strong>b1c3</strong></div>
    </div>
    <div class="chess-card">
      <div class="fen">r1bq1rk1/ppp2ppp/2n1pn2/3p4/3P4/2P1PN2/P1Q2PPP/R1B1KB1R w KQ - 0 8</div>
      <div class="prediction">Predicted move: <strong>f1d3</strong></div>
    </div>
    <div class="chess-card">
      <div class="fen">8/8/8/4k3/8/4P3/4K3/8 w - - 0 30</div>
      <div class="prediction">Predicted move: <strong>e2f3</strong></div>
    </div>
    <div class="chess-card">
      <div class="fen">8/8/8/4R3/4k3/8/4K3/8 w - - 0 50</div>
      <div class="prediction">Predicted move: <strong>e5e3</strong></div>
    </div>
    <div class="chess-card">
      <div class="fen">r1bq1rk1/pp1n1ppp/2p1pn2/3p4/3P1B2/2N1PN2/PPQ2PPP/R3KB1R b KQ - 2 10</div>
      <div class="prediction">Predicted move: <strong>f8e8</strong></div>
    </div>
  </div>
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

.chess-ai-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 15px;
  margin-top: 20px;
}

.chess-card {
  background-color: #fdfdfd;
  border: 1px solid #ddd;
  border-radius: 10px;
  padding: 15px;
  box-shadow: 0 2px 5px rgba(0,0,0,0.1);
  transition: transform 0.2s, box-shadow 0.2s;
}

.chess-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 5px 15px rgba(0,0,0,0.15);
}

.fen {
  font-family: monospace;
  font-size: 0.9rem;
  margin-bottom: 10px;
  color: #34495e;
  word-break: break-word;
}

.prediction {
  font-size: 1rem;
  color: #27ae60;
}
</style>

