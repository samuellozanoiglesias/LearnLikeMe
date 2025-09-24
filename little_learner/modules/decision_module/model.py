"""Model definition for the decision module."""

import jax.numpy as jnp
from ..extractor_modules.models import CarryModel, UnitModel


def decision_model(params: dict, x: jnp.ndarray, unit_module: dict, carry_module: dict) -> tuple:
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
            # Prepare input for the position pair
            units_input = jnp.array(x[:, [i, j]])
            
            # Get predictions from pre-trained models
            unit_output = UnitModel().apply({'params': unit_module}, units_input)
            carry_output = CarryModel().apply({'params': carry_module}, units_input)
            
            # Store predictions
            unit_val[f'{i}_{j}'] = jnp.argmax(unit_output, axis=-1)
            carry_val[f'{i}_{j}'] = jnp.argmax(carry_output, axis=-1)
    
    # Compute tens digit output using learned parameters
    salida_1 = sum(
        params[f'v_0_1_{i}_{j}'] * carry_val[f'{i}_{j}'] +
        params[f'v_1_1_{i}_{j}'] * unit_val[f'{i}_{j}']
        for i in [0, 1] for j in [2, 3]
    )

    # Compute units digit output using learned parameters
    salida_2 = sum(
        params[f'v_0_2_{i}_{j}'] * carry_val[f'{i}_{j}'] +
        params[f'v_1_2_{i}_{j}'] * unit_val[f'{i}_{j}']
        for i in [0, 1] for j in [2, 3]
    )

    return salida_1, salida_2
