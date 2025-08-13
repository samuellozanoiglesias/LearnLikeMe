import jax
import jax.numpy as jnp
import functools
import optax
from flax.training import train_state

def load_train_state(model, learning_rate, initial_params):
    tx = optax.sgd(learning_rate)
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

@functools.partial(jax.jit, static_argnums=0)
def train_step(model, state, x, y):
    loss_fn = lambda params: compute_loss(model, params, x, y)
    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state