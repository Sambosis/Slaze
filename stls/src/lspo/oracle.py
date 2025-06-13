# -*- coding: utf-8 -*-
"""
Implements the Oracle for evaluating generated 3D models.

The Oracle is a multi-faceted reward system composed of three main components:
1.  CadQueryEngine: Executes programmatic CAD scripts to generate STL files.
2.  Slicer: Interfaces with an external slicer (e.g., PrusaSlicer) to assess
    printability metrics like manifoldness, filament usage, and support material.
3.  PhysicsSimulator: Uses PyBullet to run a simple physics simulation to test
    the structural stability of the generated model under a predefined load.

The Oracle class coordinates these components to produce a single, composite
reward score for a given design, which is used to guide the RL agent.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cadquery as cq
import numpy as np
import pybullet as p

from lspo import config

# Configure a logger for this module
logger = logging.getLogger(__name__)


class CadQueryEngine:
    """
    Executes CadQuery Python scripts to generate 3D models.

    This class takes a string containing CadQuery commands, executes it
    safely, and exports the resulting 3D object as an STL file.
    """

    def __init__(self, output_dir: Path):
        """
        Initializes the CadQueryEngine.

        Args:
            output_dir (Path): The directory where generated STL files will be saved.
        """
        self.output_dir: Path = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def execute_script(
        self, script_str: str, filename_prefix: str
    ) -> Optional[Path]:
        """
        Executes a CadQuery script and saves the result as an STL file.

        The script string is expected to generate a CadQuery Workplane or Shape
        object and assign it to a variable named 'result'.

        Args:
            script_str (str): A string containing the Python code for generating
                a CadQuery model.
            filename_prefix (str): A unique prefix for the output STL file.

        Returns:
            Optional[Path]: The path to the generated STL file if successful,
                            otherwise None.
        """
        local_scope = {}
        global_scope = {"cq": cq}
        output_path = self.output_dir / f"{filename_prefix}.stl"

        try:
            exec(script_str, global_scope, local_scope)
            result_obj = local_scope.get("result")

            if result_obj is None:
                logger.warning(
                    f"Script for '{filename_prefix}' executed but 'result' object not found."
                )
                return None

            if not isinstance(result_obj, (cq.Workplane, cq.Shape)):
                logger.warning(
                    f"The 'result' object for '{filename_prefix}' is not a valid "
                    f"CadQuery Workplane or Shape. Type: {type(result_obj)}"
                )
                return None

            # Export the model
            cq.exporters.export(result_obj, str(output_path))
            logger.info(f"Successfully generated STL: {output_path}")
            return output_path

        except Exception:
            logger.error(
                f"Failed to execute or export CadQuery script for '{filename_prefix}'. "
                f"Traceback:\n{traceback.format_exc()}"
            )
            # Clean up potentially partially written file
            if output_path.exists():
                os.remove(output_path)
            return None


class Slicer:
    """
    Interfaces with a command-line slicer to analyze STL files.

    This class invokes an external slicer program (like PrusaSlicer) to
    check if a model is manifold (watertight) and to extract metrics such
    as estimated filament usage and support material.
    """

    def __init__(
        self,
        slicer_path: Optional[str],
        slicer_config_path: Optional[str],
        timeout: int,
    ):
        """
        Initializes the Slicer interface.

        Args:
            slicer_path (Optional[str]): The absolute path to the slicer executable.
            slicer_config_path (Optional[str]): The path to a slicer configuration
                file (.ini for PrusaSlicer).
            timeout (int): Timeout in seconds for the slicer process.
        """
        if not slicer_path or not Path(slicer_path).is_file():
            raise FileNotFoundError(f"Slicer executable not found at: {slicer_path}")
        self.slicer_path = Path(slicer_path)
        self.slicer_config_path = Path(slicer_config_path) if slicer_config_path else None
        self.timeout = timeout

    def analyze_stl(self, stl_path: Path) -> Dict[str, Union[bool, float, None]]:
        """
        Analyzes an STL file using the configured slicer.

        Runs the slicer in a subprocess to generate G-code and parse its
        output to extract key printability metrics.

        Args:
            stl_path (Path): The path to the STL file to be analyzed.

        Returns:
            Dict[str, Union[bool, float, None]]: A dictionary containing slicer
                metrics: 'is_manifold', 'filament_usage_mm3', 'support_material_mm3'.
                Values may be None if parsing fails.
        """
        results: Dict[str, Union[bool, float, None]] = {
            "is_manifold": None,
            "filament_usage_mm3": None,
            "support_material_mm3": None,
        }
        gcode_path = stl_path.with_suffix(".gcode")

        command = [
            str(self.slicer_path),
            "--export-gcode",
            "--gcode-comments", # Ensure comments are included for parsing
            "-o", str(gcode_path),
        ]
        if self.slicer_config_path and self.slicer_config_path.is_file():
            command.extend(["--load", str(self.slicer_config_path)])
        command.append(str(stl_path))

        try:
            process = subprocess.run(
                command, capture_output=True, text=True, timeout=self.timeout
            )

            # PrusaSlicer reports non-manifold errors to stderr
            if "is not manifold" in process.stderr:
                results["is_manifold"] = False
            else:
                # If no explicit error, assume manifold, but proceed to check G-code
                results["is_manifold"] = True

            if process.returncode != 0:
                logger.warning(
                    f"Slicer exited with code {process.returncode} for {stl_path}.\n"
                    f"Stderr: {process.stderr}"
                )
                # If slicing failed, we can't get filament data.
                return results

            if gcode_path.exists():
                with open(gcode_path, "r", encoding="utf-8") as f:
                    gcode_content = f.read()
                
                # Regex to parse filament values (in mm^3)
                filament_match = re.search(r"total filament used \[mm3\]\s*=\s*([0-9.]+)", gcode_content)
                support_match = re.search(r"total support material used \[mm3\]\s*=\s*([0-9.]+)", gcode_content)

                if filament_match:
                    results["filament_usage_mm3"] = float(filament_match.group(1))
                if support_match:
                    results["support_material_mm3"] = float(support_match.group(1))

        except subprocess.TimeoutExpired:
            logger.error(f"Slicer process timed out for {stl_path}.")
            results['is_manifold'] = False # Timeout often indicates a complex, unprintable model
        except Exception:
            logger.error(
                f"An unexpected error occurred during slicing of {stl_path}.\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            results['is_manifold'] = False
        finally:
            # Clean up temporary G-code file
            if gcode_path.exists():
                os.remove(gcode_path)
        
        return results


class PhysicsSimulator:
    """
    Uses PyBullet to perform stability analysis on a 3D model.

    This class loads an STL model into a physics simulation, applies a
    predefined load, and determines if the model remains stable.
    """
    STABILITY_THRESHOLD_POS = 0.05  # Max allowed CoM displacement in meters (5cm)
    STABILITY_THRESHOLD_ORN_Z = 0.95 # Min Z-value for the object's up-vector

    def __init__(self, sim_config: config.PhysicsSimConfig):
        """
        Initializes the PhysicsSimulator.

        Args:
            sim_config (config.PhysicsSimConfig): A dataclass containing simulation
                parameters such as gravity, load force, simulation steps, etc.
        """
        self.sim_config: config.PhysicsSimConfig = sim_config
        self._client_id: Optional[int] = None

    def _setup_simulation(self) -> None:
        """Sets up the PyBullet physics client and environment."""
        self._client_id = p.connect(p.DIRECT)
        if self._client_id < 0:
            self._client_id = None
            raise RuntimeError("Failed to connect to PyBullet.")
        p.setGravity(0, 0, -9.81, physicsClientId=self._client_id)
        p.loadURDF("plane.urdf", physicsClientId=self._client_id)

    def _teardown_simulation(self) -> None:
        """Disconnects from the PyBullet physics client."""
        if self._client_id is not None:
            p.disconnect(physicsClientId=self._client_id)
            self._client_id = None

    def test_stability(self, stl_path: Path) -> bool:
        """
        Tests the structural stability of an STL model.

        The method loads the model, applies a downward force to a specific
        region, runs the simulation, and checks if displacement or orientation
        exceeds stability thresholds.

        Args:
            stl_path (Path): The path to the STL file to be tested.

        Returns:
            bool: True if the model is deemed stable, False otherwise.
        """
        try:
            self._setup_simulation()
            
            # Load the STL as a collision shape
            try:
                collision_shape_id = p.createCollisionShape(
                    shapeType=p.GEOM_MESH,
                    fileName=str(stl_path),
                    physicsClientId=self._client_id
                )
            except p.error:
                logger.error(f"PyBullet failed to load mesh: {stl_path}")
                return False

            # Create the object on the ground, slightly elevated to avoid initial collision
            body_id = p.createMultiBody(
                baseMass=1,
                baseCollisionShapeIndex=collision_shape_id,
                basePosition=[0, 0, 0.01],
                physicsClientId=self._client_id
            )
            
            # Get initial position after settling for a moment
            for _ in range(100):
                p.stepSimulation(physicsClientId=self._client_id)
            initial_pos, _ = p.getBasePositionAndOrientation(body_id, physicsClientId=self._client_id)

            # Apply force and simulate
            force = [0, 0, -self.sim_config.applied_load]
            force_pos = self.sim_config.load_position_offset
            for _ in range(self.sim_config.sim_steps):
                p.applyExternalForce(body_id, -1, force, force_pos, p.WORLD_FRAME, physicsClientId=self._client_id)
                p.stepSimulation(physicsClientId=self._client_id)

            # Check final state for stability
            final_pos, final_orn = p.getBasePositionAndOrientation(body_id, physicsClientId=self._client_id)
            
            # Check 1: Did it move too much?
            displacement = np.linalg.norm(np.array(final_pos) - np.array(initial_pos))
            pos_stable = displacement < self.STABILITY_THRESHOLD_POS

            # Check 2: Did it tip over?
            rot_matrix = p.getMatrixFromQuaternion(final_orn, physicsClientId=self._client_id)
            up_vector_z = rot_matrix[8] # Z-component of the object's transformed Z-axis
            orn_stable = up_vector_z > self.STABILITY_THRESHOLD_ORN_Z

            is_stable = pos_stable and orn_stable
            logger.info(f"Stability test for {stl_path.name}: {'Stable' if is_stable else 'Unstable'}. Disp: {displacement:.4f}, Z-Up: {up_vector_z:.4f}")
            return is_stable

        except Exception:
            logger.error(
                f"An unexpected error occurred during physics simulation of {stl_path}.\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            return False
        finally:
            self._teardown_simulation()


class Oracle:
    """
    Coordinates evaluation components to generate a composite reward.
    """

    def __init__(self, cfg: config.Config):
        """
        Initializes the Oracle with all its components.

        Args:
            cfg (config.Config): The main project configuration object
                containing paths and parameters for all sub-components.
        """
        self.cfg: config.Config = cfg
        self.cad_engine: CadQueryEngine = CadQueryEngine(
            output_dir=self.cfg.paths.generated_stls_dir
        )
        self.slicer: Slicer = Slicer(
            slicer_path=self.cfg.paths.prusa_slicer_path,
            slicer_config_path=self.cfg.oracle.slicer.slicer_profile,
            timeout=self.cfg.oracle.slicer.timeout,
        )
        self.physics_sim: PhysicsSimulator = PhysicsSimulator(
            sim_config=self.cfg.oracle.physics_sim
        )
        self.reward_weights: config.RewardWeights = self.cfg.oracle.reward_weights

    def evaluate_design(
        self, cad_script: str, unique_id: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Performs a full evaluation of a CAD script and returns a reward.

        This is the main entry point for the Oracle. It orchestrates the
        generation, slicing, and simulation of a design to produce a final
        reward score and a dictionary of detailed metrics.

        Args:
            cad_script (str): The CadQuery script representing the design.
            unique_id (str): A unique identifier for this evaluation run,
                             used for naming output files.

        Returns:
            Tuple[float, Dict[str, Any]]: A tuple containing:
                - The final composite reward score (float).
                - A dictionary of all collected metrics for logging.
        """
        metrics: Dict[str, Any] = {"cad_script": cad_script, "id": unique_id}

        # 1. Generate STL file from script
        stl_path = self.cad_engine.execute_script(cad_script, unique_id)
        if stl_path is None:
            metrics["generation_successful"] = False
            metrics["is_manifold"] = False
            return self._calculate_composite_reward(metrics), metrics
        metrics["generation_successful"] = True
        metrics["stl_path"] = str(stl_path)
        
        # 2. Analyze the STL with the slicer
        slicer_metrics = self.slicer.analyze_stl(stl_path)
        metrics.update(slicer_metrics)
        
        if not metrics.get("is_manifold"):
            # Non-manifold objects are not testable and get a failure penalty
            return self._calculate_composite_reward(metrics), metrics
        
        # 3. Test structural stability
        is_stable = self.physics_sim.test_stability(stl_path)
        metrics["is_stable"] = is_stable

        # 4. Calculate final composite reward
        reward = self._calculate_composite_reward(metrics)

        return reward, metrics

    def _calculate_composite_reward(self, metrics: Dict[str, Any]) -> float:
        """
        Calculates a final reward score from a dictionary of metrics.

        The calculation is based on a weighted sum of the different metric
        components, with weights defined in the project configuration.

        Args:
            metrics (Dict[str, Any]): A dictionary containing raw metrics from
                the slicer and simulator (e.g., 'is_manifold', 'filament_usage_mm3',
                'is_stable').

        Returns:
            float: The calculated composite reward score.
        """
        # Handle catastrophic failure first
        if not metrics.get("is_manifold", False):
            return self.reward_weights.printability_failure

        reward = 0.0
        
        # Base reward for being printable
        reward += self.reward_weights.printability_success
        
        # Bonus for being stable
        if metrics.get("is_stable", False):
            reward += self.reward_weights.stability_weight

        # Penalty for material usage
        filament_usage = metrics.get("filament_usage_mm3")
        if filament_usage is not None:
            reward += filament_usage * self.reward_weights.filament_usage_weight

        # Penalty for support material usage
        support_material = metrics.get("support_material_mm3")
        if support_material is not None:
             reward += support_material * self.reward_weights.support_material_weight
        
        return reward