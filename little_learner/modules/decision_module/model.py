import jax.numpy as jnp
import jax.nn
from ..extractor_modules.models import ExtractorModel, CarryModel, UnitModel

"""Model definition for the decision module."""

def decision_model_vector(params: dict, x: jnp.ndarray, 
                          unit_module: dict, carry_module: dict,
                          unit_structure=[256, 128], carry_structure=[16]) -> tuple:
    """
    Forward pass of the decision module using probability vectors from submodules.
    Instead of argmax, uses softmax to get probability distributions for each digit.
    The next layer uses these probability vectors for output computation.
    Returns probability vectors for tens and units.
    """
    number_size = x.shape[1] // 2
    idx_i = jnp.arange(number_size)
    idx_j = jnp.arange(number_size, 2 * number_size)
    pairs = jnp.array([(i, j) for i in idx_i for j in idx_j])
    single_digit_inputs = x[:, pairs].reshape(x.shape[0], -1, 2)
    carry_outputs = jnp.stack([
        jax.nn.softmax(ExtractorModel(structure=carry_structure, output_dim=2).apply({'params': carry_module}, single_digit_inputs[:, k]), axis=-1)
        for k in range(single_digit_inputs.shape[1])
    ], axis=1)
    unit_outputs = jnp.stack([
        jax.nn.softmax(ExtractorModel(structure=unit_structure, output_dim=10).apply({'params': unit_module}, single_digit_inputs[:, k]), axis=-1)
        for k in range(single_digit_inputs.shape[1])
    ], axis=1)
    # Flatten probability vectors for each pair
    carry_flat = carry_outputs.reshape(x.shape[0], -1)
    unit_flat = unit_outputs.reshape(x.shape[0], -1)
    concat_features = jnp.concatenate([carry_flat, unit_flat], axis=1)
    outputs = jnp.stack([
        jnp.dot(concat_features, params[f'dense_{i}']) for i in range(number_size + 1)
    ], axis=1)
    return outputs


def decision_model_argmax(params: dict, x: jnp.ndarray, 
                          unit_module: dict, carry_module: dict,
                          unit_structure=[256, 128], carry_structure=[16]) -> tuple:
    """
    Forward pass of the decision module.
    
    Args:
        params: Dictionary containing the trainable parameters
        x: Input tensor of shape (batch_size, 4) containing [tens1, units1, tens2, units2]
        unit_module: Pre-trained unit extraction model parameters
        carry_module: Pre-trained carry detection model parameters
        
    Returns:
        Tuple (tens_out, units_out) containing predicted tens and units of the sum
    """
    number_size = x.shape[1] // 2
    # Vectorized feature extraction for all digit pairs
    idx_i = jnp.arange(number_size)
    idx_j = jnp.arange(number_size, 2 * number_size)
    # Create all (i, j) pairs
    pairs = jnp.array([(i, j) for i in idx_i for j in idx_j])
    # Gather inputs for all pairs: shape (num_pairs, batch, 2)
    single_digit_inputs = x[:, pairs].reshape(x.shape[0], -1, 2)
    # Vectorized apply for carry and unit models
    carry_outputs = jnp.stack([
        jnp.argmax(ExtractorModel(structure=carry_structure, output_dim=2).apply({'params': carry_module}, single_digit_inputs[:, k]), axis=-1)
        for k in range(single_digit_inputs.shape[1])
    ], axis=1)
    unit_outputs = jnp.stack([
        jnp.argmax(ExtractorModel(structure=unit_structure, output_dim=10).apply({'params': unit_module}, single_digit_inputs[:, k]), axis=-1)
        for k in range(single_digit_inputs.shape[1])
    ], axis=1)
    # Concatenate features: shape (batch, num_pairs * 2)
    concat_features = jnp.concatenate([carry_outputs, unit_outputs], axis=1)
    # Dense layer for each digit output, stack results: shape (batch, number_size+1)
    outputs = jnp.stack([
        jnp.dot(concat_features, params[f'dense_{i}']) for i in range(number_size + 1)
    ], axis=1)
    # Return shape (batch, number_size+1) for compatibility with compute_loss
    return outputs


# -------------------- Legacy Decision Module ---------------- #

def OLD_decision_model_vector(params: dict, x: jnp.ndarray, unit_module: dict, carry_module: dict,
                         unit_hidden1=256, unit_hidden2=128, unit_output_dim=10, carry_hidden1=16, carry_output_dim=2) -> tuple:
    """
    Forward pass of the decision module using probability vectors from submodules.
    Instead of argmax, uses softmax to get probability distributions for each digit.
    The next layer uses these probability vectors for output computation.
    Returns probability vectors for tens and units.
    """
    carry_val = {}
    unit_val = {}

    # Extract features for each digit pair using pre-trained models
    for i in [0, 1]:  # Position of first number (tens/units)
        for j in [2, 3]:  # Position of second number (tens/units)
            units_input = jnp.array(x[:, [i, j]])
            unit_output = UnitModel(hidden1=unit_hidden1, hidden2=unit_hidden2, output_dim=unit_output_dim).apply({'params': unit_module}, units_input)
            carry_output = CarryModel(hidden1=carry_hidden1, output_dim=carry_output_dim).apply({'params': carry_module}, units_input)
            # Convert to probability vectors using softmax
            unit_val[f'{i}_{j}'] = jax.nn.softmax(unit_output, axis=-1)
            carry_val[f'{i}_{j}'] = jax.nn.softmax(carry_output, axis=-1)
            carry_val[f'{i}_{j}'] = jnp.pad(carry_val[f'{i}_{j}'], ((0, 0), (0, 8)), mode='constant')

    # Compute tens digit output using learned parameters
    tens_output = sum(
        params[f'v_0_1_{i}_{j}'] * carry_val[f'{i}_{j}'] +
        params[f'v_1_1_{i}_{j}'] * unit_val[f'{i}_{j}']
        for i in [0, 1] for j in [2, 3]
    )

    # Compute units digit output using learned parameters
    units_output = sum(
        params[f'v_0_2_{i}_{j}'] * carry_val[f'{i}_{j}'] +
        params[f'v_1_2_{i}_{j}'] * unit_val[f'{i}_{j}']
        for i in [0, 1] for j in [2, 3]
    )

    return tens_output, units_output


def OLD_decision_model_argmax(params: dict, x: jnp.ndarray, unit_module: dict, carry_module: dict,
                         unit_hidden1=256, unit_hidden2=128, unit_output_dim=10, carry_hidden1=16, carry_output_dim=2) -> tuple:
    """
    Forward pass of the decision module.
    
    Args:
        params: Dictionary containing the trainable parameters
        x: Input tensor of shape (batch_size, 4) containing [tens1, units1, tens2, units2]
        unit_module: Pre-trained unit extraction model parameters
        carry_module: Pre-trained carry detection model parameters
        
    Returns:
        Tuple (tens_out, units_out) containing predicted tens and units of the sum
    """
    carry_val = {}
    unit_val = {}

    # Extract features for each digit pair using pre-trained models
    for i in [0, 1]:  # Position of first number (tens/units)
        for j in [2, 3]:  # Position of second number (tens/units)
            units_input = jnp.array(x[:, [i, j]])
            unit_output = UnitModel(hidden1=unit_hidden1, hidden2=unit_hidden2, output_dim=unit_output_dim).apply({'params': unit_module}, units_input)
            carry_output = CarryModel(hidden1=carry_hidden1, output_dim=carry_output_dim).apply({'params': carry_module}, units_input)
            unit_val_argmax = jnp.argmax(unit_output, axis=-1)
            carry_val_argmax = jnp.argmax(carry_output, axis=-1)
            unit_val[f'{i}_{j}'] = unit_val_argmax
            carry_val[f'{i}_{j}'] = carry_val_argmax
    
    # Compute tens digit output using learned parameters
    tens_output = sum(
        params[f'v_0_1_{i}_{j}'] * carry_val[f'{i}_{j}'] +
        params[f'v_1_1_{i}_{j}'] * unit_val[f'{i}_{j}']
        for i in [0, 1] for j in [2, 3]
    )

    units_output = sum(
        params[f'v_0_2_{i}_{j}'] * carry_val[f'{i}_{j}'] +
        params[f'v_1_2_{i}_{j}'] * unit_val[f'{i}_{j}']
        for i in [0, 1] for j in [2, 3]
    )

    return tens_output, units_output
