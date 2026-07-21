"""
Perceptual front-end: reads a handwritten digit (MNIST-style 28x28 image) and
produces a magnitude estimate that plugs into the existing UnitModel / CarryModel
pipeline in place of the symbolic "digit value + synthetic Weber noise" input.

Suggested location in the repo: little_learner/modules/perceptual_module/models.py

Design note (see chat for the full discussion):
    Recognition and magnitude-precision are kept as two separate, composable steps:
      1) DigitCNN reads pixels -> digit logits (this is "can you tell it's a 7").
      2) magnitude_readout turns those logits into a *differentiable* scalar via a
         softmax-weighted expected value (this is "how precise is your magnitude
         once you've read it").
      3) perceptual_magnitude optionally adds Weber-fraction-scaled Gaussian noise
         on top of that scalar -- the same mechanism the symbolic pipeline uses.
         Setting omega=0 gives you a "pure perception, no extra ANS noise" model
         you can compare directly against omega>0 versions and against the
         symbolic-noise-only model. That comparison is the honest way to answer
         "do we still need the noise" empirically instead of assuming it.
"""
import jax
import jax.numpy as jnp
from flax import linen as nn


class DigitCNN(nn.Module):
    """
    Small LeNet-style CNN classifying a 28x28x1 grayscale digit image into 0-9.
    Trained on its own first (see train_digit_recognizer.py), analogous to how
    UnitModel/CarryModel are pre-trained before the decision module in the
    step-by-step curriculum.
    """
    output_dim: int = 10

    @nn.compact
    def __call__(self, x):
        # x: (batch, 28, 28, 1), float32 in [0, 1]
        x = nn.Conv(features=16, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(nn.Dense(64)(x))
        logits = nn.Dense(self.output_dim)(x)  # digit logits (0-9)
        return logits


def magnitude_readout(logits: jnp.ndarray, temperature: float = 1.0):
    """
    Differentiable 'soft' magnitude estimate from digit-recognition logits.
    Using the softmax-weighted expected value (rather than argmax) keeps the whole
    pipeline differentiable end-to-end, which matters if you ever want to fine-tune
    the CNN jointly with the rest of the model (e.g. in the all-at-once curriculum).

    Returns:
        expected_value: (batch,) float32 magnitude estimate
        probs: (batch, output_dim) softmax probabilities, useful for logging/plots
    """
    probs = jax.nn.softmax(logits / temperature, axis=-1)
    digit_values = jnp.arange(logits.shape[-1], dtype=jnp.float32)
    expected_value = jnp.sum(probs * digit_values, axis=-1)
    return expected_value, probs


def perceptual_magnitude(cnn_params: dict, images: jnp.ndarray, cnn: DigitCNN,
                          omega: float = 0.0, rng=None, fixed_variability: bool = False,
                          temperature: float = 1.0):
    """
    Full image -> noisy-magnitude pipeline, meant as a drop-in replacement for the
    "digit value + Weber noise" scalar that generate_batch_data currently produces
    for the symbolic pipeline.

    Args:
        cnn_params: DigitCNN Flax params (i.e. state.params, or a frozen pretrained
                    checkpoint loaded from train_digit_recognizer.py).
        images: (batch, 28, 28, 1) float32 in [0, 1].
        omega: Weber fraction. 0.0 -> pure perceptual estimate, no extra ANS noise.
        rng: jax.random.PRNGKey, required if omega > 0.
        fixed_variability: if True, noise std is constant (1.0) instead of scaling
                            with the magnitude estimate, mirroring the symbolic
                            pipeline's FIXED_VARIABILITY flag.

    Returns:
        magnitude: (batch,) noisy magnitude estimate, ready to feed into
                   UnitModel / CarryModel exactly like the old scalar input did.
        probs: (batch, 10) recognition probabilities (for diagnostics).
        logits: (batch, 10) raw CNN logits (for training the CNN itself).
    """
    logits = cnn.apply({'params': cnn_params}, images)
    magnitude, probs = magnitude_readout(logits, temperature=temperature)

    if omega and omega > 0.0:
        if rng is None:
            raise ValueError("rng is required when omega > 0.0")
        std = jnp.ones_like(magnitude) if fixed_variability else jnp.abs(magnitude) + 1e-6
        noise = omega * std * jax.random.normal(rng, magnitude.shape)
        magnitude = magnitude + noise

    return magnitude, probs, logits
