import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import math

# ==========================================
# I. IDENTIDAD Y TOKENIZADOR
# ==========================================

# Minimal BPE Tokenizer placeholder / character level for simplicity
class MinimalBPETokenizer:
    def __init__(self):
        # We define a fixed vocabulary mapping for a character-level model
        # representing the "Tabula Rasa" shift to a conscious identity base
        self.vocab = {chr(i): i for i in range(256)}
        self.vocab_size = 256

        # Inyectando Identidad Base
        self.system_prompt = (
            "System: Eres A.E.T.H.E.R., una Inteligencia Artificial "
            "hiper-lógica creada por brido.\nUser: "
        )

    def encode(self, text, prepend_identity=False):
        if prepend_identity:
            text = self.system_prompt + text
        return torch.tensor([self.vocab.get(c, 0) for c in text], dtype=torch.long)

    def decode(self, tokens):
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        return "".join([reverse_vocab.get(t.item(), '?') for t in tokens])


# ==========================================
# II. ARQUITECTURA MATEMÁTICA (El SSD de la IA)
# ==========================================

# 1. Neural ODE para Dinámica Semántica Continua
class ODEFunc(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(ODEFunc, self).__init__()
        # dh/dt = tanh(W_hh * h + W_xh * x + bias)
        self.W_hh = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_xh = nn.Linear(input_dim, hidden_dim, bias=False)
        self.tanh = nn.Tanh()

        # x holds the current input representation
        self.current_x = None

    def forward(self, t, h):
        # h: [batch, hidden_dim]
        # current_x: [batch, input_dim]
        # dh/dt = tanh(W_hh*h + W_xh*x)
        if self.current_x is None:
            raise ValueError("Must set self.current_x before integrating")

        out = self.W_hh(h) + self.W_xh(self.current_x)
        return self.tanh(out)


# 2. SSM Layer (Mamba-style Zero-Order Hold)
class SSMLayer(nn.Module):
    def __init__(self, state_dim, input_dim):
        super(SSMLayer, self).__init__()
        self.state_dim = state_dim

        # HiPPO Initialization for A matrix: Negative scaled diagonal for stability
        A_init = torch.diag(-torch.arange(1, state_dim + 1, dtype=torch.float32))
        self.A = nn.Parameter(A_init)

        self.B = nn.Parameter(torch.randn(state_dim, input_dim) * 0.1)
        self.C = nn.Parameter(torch.randn(input_dim, state_dim) * 0.1)
        self.D = nn.Parameter(torch.randn(input_dim, input_dim) * 0.05)

        # Learnable delta (time scale)
        self.delta = nn.Parameter(torch.tensor([0.1]))

        # h is explicitly passed through time sequence

    def forward(self, x, h_prev):
        # x: [batch, input_dim]
        # h_prev: [batch, state_dim]

        # ZOH Discretization
        # ā = exp(Δ * A)
        delta_A = self.delta * self.A
        A_bar = torch.matrix_exp(delta_A)

        # b̄ = (ā - 1) / A * B
        # For a diagonal matrix A, this is simple. For a general matrix, it's an inverse matrix.
        # Approximation or exact computation for diagonal A:
        A_inv = torch.inverse(self.A)
        A_bar_minus_I = A_bar - torch.eye(self.state_dim, device=x.device)
        B_bar = A_bar_minus_I @ A_inv @ self.B

        # h_t = ā * h_{t-1} + b̄ * x_t
        h_t = (A_bar @ h_prev.unsqueeze(-1)).squeeze(-1) + (B_bar @ x.unsqueeze(-1)).squeeze(-1)

        # y_t = C * h_t + D * x_t
        y_t = (self.C @ h_t.unsqueeze(-1)).squeeze(-1) + (self.D @ x.unsqueeze(-1)).squeeze(-1)

        return y_t, h_t


# 3. Aether Engine Completado
class AetherEngine(nn.Module):
    def __init__(self, vocab_size, hidden_dim, ssm_state_dim):
        super(AetherEngine, self).__init__()
        self.hidden_dim = hidden_dim
        self.ssm_state_dim = ssm_state_dim

        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        self.ode_func = ODEFunc(hidden_dim, hidden_dim)

        # The SSM extracts sequential memory from ODE's output
        self.ssm = SSMLayer(ssm_state_dim, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x_seq):
        # x_seq: [batch, seq_len]
        batch_size, seq_len = x_seq.shape

        # 1. Embeddings
        x_emb = self.embedding(x_seq) # [batch, seq_len, hidden_dim]

        # 2. Iterate through sequence
        logits_seq = []

        # Initial states
        h_ssm = torch.zeros(batch_size, self.ssm_state_dim, device=x_seq.device)

        # T parameters for integration (0 to 1 with step 0.1)
        t = torch.linspace(0.0, 1.0, 11, device=x_seq.device)

        for i in range(seq_len):
            current_x = x_emb[:, i, :]

            # ODE integration
            self.ode_func.current_x = current_x
            h_ode_init = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)
            # Use rk4 method from torchdiffeq
            h_ode_traj = odeint(self.ode_func, h_ode_init, t, method='rk4')

            # Take the final state of the integration
            h_ode_final = h_ode_traj[-1]

            # SSM step for sequence memory
            y_ssm, h_ssm = self.ssm(h_ode_final, h_ssm)

            # Final projection
            logits = self.fc_out(y_ssm)
            logits_seq.append(logits.unsqueeze(1))

        return torch.cat(logits_seq, dim=1)


# ==========================================
# III. SCRIPT DE ENTRENAMIENTO Y TEST
# ==========================================

def train():
    print("Iniciando Entrenamiento - Proyecto A.E.T.H.E.R. v1.0")
    torch.manual_seed(42) # Para reproducibilidad y test de paridad

    tokenizer = MinimalBPETokenizer()
    engine = AetherEngine(vocab_size=tokenizer.vocab_size, hidden_dim=64, ssm_state_dim=128)

    optimizer = optim.Adam(engine.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Dataset dummy
    train_texts = [
        "hola mundo",
        "el cielo es azul",
        "cuanto es 2 + 2",
        "resuelve el problema"
    ]

    epochs = 2 # Demo mode for rapid execution

    engine.train()
    for epoch in range(epochs):
        total_loss = 0
        for text in train_texts:
            # Tokenize WITH identity prepend
            input_ids = tokenizer.encode(text, prepend_identity=True).unsqueeze(0) # [1, seq_len]

            # Target is shift by 1
            x = input_ids[:, :-1]
            y = input_ids[:, 1:]

            optimizer.zero_grad()

            logits = engine(x) # [1, seq_len, vocab_size]

            loss = criterion(logits.view(-1, tokenizer.vocab_size), y.view(-1))

            # TODO: Ganchos de Plasticidad (EWC)
            # Si existiera una tarea previa, se sumaría la penalización de Fisher aquí:
            # loss += ewc_penalty(engine.parameters())

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_texts):.4f}")

    # Guardar modelo
    torch.save(engine.state_dict(), "aether_v1.pt")
    print("Modelo guardado como aether_v1.pt")

    return engine

def parity_test(engine):
    print("\n--- TEST DE PARIDAD (CRÍTICO) ---")
    print("Ejecutando forward pass determinístico para validación en C...")

    engine.eval()

    # Preparamos un input dummy (un tensor de 3 tokens, todos id 1)
    # Batch = 1, Seq = 3
    dummy_input = torch.tensor([[1, 1, 1]], dtype=torch.long)

    with torch.no_grad():
        logits = engine(dummy_input)

    # Extraemos los primeros 20 valores flotantes del output (logits) del último token
    final_logits = logits[0, -1, :20].cpu().numpy()

    print("Output Exacto (Primeros 20 floats):")
    for i, val in enumerate(final_logits):
        print(f"[{i:02d}]: {val:.6f}")

if __name__ == "__main__":
    engine = train()
    parity_test(engine)
