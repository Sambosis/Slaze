# -*- coding: utf-8 -*-
"""
Calculates the final composite reward for a generated 3D model.

This module provides the core function for evaluating the quality of a generated
design. It integrates feedback from various "oracles" (slicing, physics)
and combines them into a single scalar reward signal, which is essential for
the reinforcement learning agent's training loop. The weighting of different
reward components is managed through the central project configuration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Tuple

from src.lspo_3d import config

# Type checking to avoid circular imports, standard for complex projects.
if TYPE_CHECKING:
    from lspo_3d.oracles.physics import PhysicsOracle
    from lspo_3d.oracles.slicer import SlicerOracle

logger = logging.getLogger(__name__)


def calculate_reward(
    stl_file_path: str,
    slicer_oracle: "SlicerOracle",
    physics_oracle: "PhysicsOracle",
) -> Tuple[float, Dict[str, float]]:
    """
    Calculates a composite reward for a generated 3D model.

    This function orchestrates the evaluation of a given .stl file by calling
    the SlicerOracle and PhysicsOracle. It then combines their outputs into a
    single scalar reward value using a weighted sum, with weights defined in
    the project's config file.

    The function is designed to be the final step in the evaluation pipeline,
    translating multiple complex metrics into a single signal for the RL agent.

    Args:
        stl_file_path (str): The absolute path to the .stl file to evaluate.
        slicer_oracle (SlicerOracle): An instantiated SlicerOracle for
            evaluating printability metrics.
        physics_oracle (PhysicsOracle): An instantiated PhysicsOracle for
            evaluating physical stability.

    Returns:
        Tuple[float, Dict[str, float]]: A tuple containing:
        - The final composite reward (float), suitable for the RL agent.
        - A dictionary detailing the component scores and the raw values from
          the oracles for logging and analysis (e.g.,
          {'reward_total': 0.85, 'reward_printability': 1.0, ...
           'raw_slicing_success': 1.0, 'raw_support_volume': 15.3, ...}).
    """
    # 0. Initialize and prepare
    stl_path = Path(stl_file_path)
    detailed_results: Dict[str, float] = {}
    reward_weights = config.REWARD_WEIGHTS
    reward_norm = config.REWARD_NORMALIZATION

    # 1. Invoke the SlicerOracle to get printability metrics.
    slicer_metrics = slicer_oracle.slice_and_evaluate(stl_path)
    detailed_results['raw_slicing_success'] = 0.0

    # 2. If slicing failed, assign a large penalty and return immediately.
    if slicer_metrics is None or not slicer_metrics.is_sliceable:
        logger.warning(f"Slicing failed for {stl_path.name}. Assigning failure penalty.")
        total_reward = reward_weights.get('slicing_failure_penalty', -10.0)
        detailed_results.update({
            'reward_total': total_reward,
            'reward_printability': total_reward,
            'reward_support': 0.0,
            'reward_efficiency': 0.0,
            'reward_stability': 0.0,
            'raw_support_volume': -1.0,
            'raw_filament_volume': -1.0,
            'raw_stability_score': 0.0,
        })
        return total_reward, detailed_results

    # If slicing succeeded, populate raw slicer metrics
    detailed_results['raw_slicing_success'] = 1.0
    detailed_results['raw_support_volume'] = slicer_metrics.support_material_volume_mm3
    detailed_results['raw_filament_volume'] = slicer_metrics.total_filament_volume_mm3
    detailed_results['raw_print_time_s'] = slicer_metrics.estimated_print_time_s

    # 3. Invoke the PhysicsOracle to get the stability score.
    try:
        stability_score = physics_oracle.run_simulation(stl_path)
        detailed_results['raw_stability_score'] = stability_score
    except Exception as e:
        logger.error(f"Physics simulation crashed for {stl_path.name}: {e}")
        stability_score = 0.0 # Treat simulation crash as instability
        detailed_results['raw_stability_score'] = stability_score

    # 4. & 5. Calculate individual reward components based on oracle outputs and weights.
    
    # Printability Reward: A fixed bonus for successfully slicing the model.
    printability_reward = reward_weights.get('printability_success_bonus', 1.0)
    detailed_results['reward_printability'] = printability_reward

    # Support Penalty: Negative reward, proportional to support material volume.
    # The penalty increases as more support is needed.
    support_penalty = (
        -slicer_metrics.support_material_volume_mm3 *
        reward_weights.get('support_penalty_factor', 0.01)
    )
    detailed_results['reward_support'] = support_penalty

    # Material Efficiency Reward: Inversely proportional to filament usage.
    # Normalized to prevent reward explosion for tiny models.
    # The reward approaches the bonus value as volume -> 0.
    ref_volume = reward_norm.get('reference_volume', 10000.0) # Default to 100cm^3
    efficiency_bonus = reward_weights.get('material_efficiency_bonus', 1.0)
    efficiency_reward = efficiency_bonus / (1 + slicer_metrics.total_filament_volume_mm3 / ref_volume)
    detailed_results['reward_efficiency'] = efficiency_reward

    # Stability Reward: A fixed bonus if the model is stable.
    stability_reward = stability_score * reward_weights.get('stability_success_bonus', 2.0)
    detailed_results['reward_stability'] = stability_reward

    # 6. Sum the individual components to compute the final, total reward.
    total_reward = (
        printability_reward +
        support_penalty +
        efficiency_reward +
        stability_reward
    )
    detailed_results['reward_total'] = total_reward
    
    logger.debug(f"Evaluation for {stl_path.name}: Total Reward = {total_reward:.4f}, Details = {detailed_results}")

    # 7. Return the total reward and the detailed dictionary of components.
    return total_reward, detailed_results