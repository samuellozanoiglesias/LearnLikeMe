import jax
import jax.numpy as jnp
from flax import linen as nn

class ExtractorModel(nn.Module):
    structure: list
    output_dim: int
    @nn.compact
    def __call__(self, x):
        for layer in self.structure:
            x = nn.relu(nn.Dense(layer)(x))
        x = nn.Dense(self.output_dim)(x)  # logits (0-9)
        return x

# --------------------- Feedforward Models -------------------
class CarryModel(nn.Module):
    hidden1: int = 16
    output_dim: int = 2
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(self.hidden1)(x))
        x = nn.Dense(self.output_dim)(x)             # raw logits
        return x


class UnitModel(nn.Module):
    hidden1: int = 256
    hidden2: int = 128
    output_dim: int = 10
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(self.hidden1)(x))    # ancha para absorber el ruido
        x = nn.relu(nn.Dense(self.hidden2)(x))
        x = nn.Dense(self.output_dim)(x)     # logits (0-9)
        return x

# -------------------- Legacy - LSTM Models -------------------
class CarryLSTMModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        lstm_1 = nn.LSTMCell(features=16)
        dense = nn.Dense(2)
        carry1 = lstm_1.initialize_carry(jax.random.PRNGKey(0), (x.shape[0],))
        for t in range(x.shape[1]):
            carry1, x_t = lstm_1(carry1, x[:, t])
        hidden_state = carry1[0]
        final_output = nn.softmax(dense(hidden_state))
        return final_output

class UnitLSTMModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        lstm_1 = nn.LSTMCell(features=16)
        lstm_2 = nn.LSTMCell(features=32)
        lstm_3 = nn.LSTMCell(features=16)
        dense = nn.Dense(10)
        carry1 = lstm_1.initialize_carry(jax.random.PRNGKey(0), (x.shape[0],))
        carry2 = lstm_2.initialize_carry(jax.random.PRNGKey(1), (x.shape[0],))
        carry3 = lstm_3.initialize_carry(jax.random.PRNGKey(2), (x.shape[0],))
        for t in range(x.shape[1]):
            carry1, x_t = lstm_1(carry1, x[:, t])
            carry2, x_t = lstm_2(carry2, x_t)
            carry3, x_t = lstm_3(carry3, x_t)
        hidden_state = carry2[0]
        final_output = nn.softmax(dense(hidden_state))
        return final_output
