import jax
import jax.numpy as jnp
from flax import linen as nn

class CarryModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(16)(x)) 
        x = nn.Dense(2)(x)             # raw logits
        return x

class UnitModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.tanh(nn.Dense(32)(x))
        x = nn.tanh(nn.Dense(16)(x))
        x = nn.Dense(10)(x)  # logits
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
