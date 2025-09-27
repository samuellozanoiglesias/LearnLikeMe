import jax
import jax.numpy as jnp
import functools
import optax
from flax.training import train_state

def load_train_state(model, learning_rate, initial_params):
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=initial_params, tx=tx)

def compute_loss(model, params, x, y):
    logits = model.apply({"params": params}, x)
    loss = optax.softmax_cross_entropy(logits, y).mean()
    return loss

def evaluate(model, params, x, y):
    logits = model.apply({"params": params}, x)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == jnp.argmax(y, axis=-1))
    return accuracy

def get_predictions(model, state, x, y):
    logits = model.apply({"params": state.params}, x)
    predictions = jnp.argmax(logits, axis=-1)
    true_labels = jnp.argmax(y, axis=-1)
    return predictions, true_labels

@jax.jit
def train_step(state, x, y):
    # Use the apply_fn stored in the TrainState to avoid passing Flax Module instances
    loss_fn = lambda params: optax.softmax_cross_entropy(state.apply_fn({"params": params}, x), y).mean()
    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, grads