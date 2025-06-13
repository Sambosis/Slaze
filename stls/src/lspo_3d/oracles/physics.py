# -*- coding: utf-8 -*-
"""
A wrapper for the PyBullet physics engine to evaluate the stability of 3D models.

This module provides the PhysicsOracle class, which is responsible for loading an
.stl file into a PyBullet simulation, running a standardized stability test,
and returning a score based on the outcome.
"""

from pathlib import Path
from typing import Dict, Any
import logging
import math

import pybullet as p

from src.lspo_3d import config

# Set up a logger for this module
logger = logging.getLogger(__name__)


class PhysicsOracle:
    """
    Manages a PyBullet simulation to assess the physical stability of a 3D model.

    This class encapsulates the setup, execution, and teardown of a PyBullet
    simulation environment. It provides a single public method to run a
    pre-defined stability test on a given .stl model.
    """

    def __init__(self, simulation_config: Dict[str, Any] = None):
        """
        Initializes the PhysicsOracle with specific simulation parameters.

        Args:
            simulation_config (Dict[str, Any], optional): A dictionary
                containing simulation settings. If None, it defaults to the
                physics settings in the project's config.py file.
                Expected keys might include 'connection_mode', 'gravity',
                'time_step', 'simulation_steps', 'stability_threshold_rad',
                'model_mass_kg', and 'model_initial_z'.
        """
        self.config = simulation_config or config.PHYSICS_ORACLE_SETTINGS
        self.client_id: int = -1

        # Extract settings from config with sane defaults
        self.connection_mode = p.DIRECT if self.config.get("connection_mode", "DIRECT").upper() == "DIRECT" else p.GUI
        self.gravity = self.config.get("gravity", [0, 0, -9.81])
        self.time_step = self.config.get("time_step", 1.0 / 240.0)
        self.simulation_steps = self.config.get("simulation_steps", 500)
        self.stability_threshold_rad = self.config.get("stability_threshold_rad", math.radians(15)) # e.g., 15 degrees
        self.model_mass_kg = self.config.get("model_mass_kg", 1.0)
        self.model_initial_z = self.config.get("model_initial_z", 0.1) # Start slightly above ground

    def _setup_simulation(self) -> None:
        """
        Initializes the PyBullet physics client and configures the environment.

        This method sets up the connection to the physics server (either GUI or
        DIRECT), sets gravity, and creates a static ground plane. It should be
        called at the beginning of each simulation run.
        """
        try:
            self.client_id = p.connect(self.connection_mode)
            p.setGravity(*self.gravity, physicsClientId=self.client_id)
            p.setTimeStep(self.time_step, physicsClientId=self.client_id)
            p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True, physicsClientId=self.client_id)
        except p.error as e:
            logger.error(f"Failed to set up PyBullet simulation: {e}")
            self.client_id = -1 # Ensure client_id reflects failed state
            raise

    def _cleanup_simulation(self) -> None:
        """
        Disconnects from the PyBullet physics server.

        This method should be called after a simulation is complete to ensure
        a clean shutdown of the PyBullet client.
        """
        if self.client_id >= 0:
            try:
                p.disconnect(physicsClientId=self.client_id)
            except p.error as e:
                logger.warning(f"Error during PyBullet disconnection: {e}")
            finally:
                self.client_id = -1

    def _load_model(self, stl_file_path: Path) -> int:
        """
        Loads a 3D model from an .stl file into the simulation.

        This method handles the loading of the mesh, setting its initial
        position and orientation on the ground plane, and potentially adjusting
        its physical properties like mass and friction.

        Args:
            stl_file_path (Path): The absolute path to the .stl file to be loaded.

        Returns:
            int: The unique ID of the loaded model body in the simulation.
        
        Raises:
            p.error: If the model cannot be loaded into the simulation.
        """
        # PyBullet uses p.GEOM_MESH to load arbitrary triangle meshes like STL.
        # It's important that the STL file is valid.
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=str(stl_file_path),
            meshScale=[1, 1, 1], # Assuming STL is in meters; adjust if needed.
            physicsClientId=self.client_id
        )
        
        if collision_shape_id < 0:
            raise p.error(f"Failed to create collision shape for {stl_file_path}")

        initial_position = [0, 0, self.model_initial_z]
        initial_orientation = p.getQuaternionFromEuler([0, 0, 0])
        
        model_id = p.createMultiBody(
            baseMass=self.model_mass_kg,
            baseCollisionShapeIndex=collision_shape_id,
            basePosition=initial_position,
            baseOrientation=initial_orientation,
            physicsClientId=self.client_id
        )

        p.changeDynamics(model_id, -1, lateralFriction=0.5, spinningFriction=0.1, rollingFriction=0.1)
        
        return model_id

    def _run_stability_test(self, model_id: int) -> bool:
        """
        Executes the core stability test on the loaded model.

        This involves simulating the physics for a set number of steps. The test
        involves checking if the model tips over under its own weight.
        Stability is determined by checking the model's final orientation against a threshold.

        Args:
            model_id (int): The unique ID of the model to be tested.

        Returns:
            bool: True if the model is deemed stable, False otherwise.
        """
        for _ in range(self.simulation_steps):
            p.stepSimulation(physicsClientId=self.client_id)

        # Get final orientation to check for tipping
        try:
            _, orientation_quat = p.getBasePositionAndOrientation(model_id, physicsClientId=self.client_id)
        except p.error:
            logger.warning(f"Could not get orientation for model ID {model_id}. Assuming instability.")
            return False

        # Convert quaternion to Euler angles (roll, pitch, yaw)
        roll, pitch, _ = p.getEulerFromQuaternion(orientation_quat, physicsClientId=self.client_id)
        
        # Check if the tilt (roll or pitch) exceeds the stability threshold
        is_stable = abs(roll) < self.stability_threshold_rad and abs(pitch) < self.stability_threshold_rad
        
        return is_stable

    def run_simulation(self, stl_file_path: Path) -> float:
        """
        Runs a full stability simulation for a given .stl file and returns a score.

        This is the main public method for the class. It orchestrates the setup,
        model loading, test execution, and cleanup for a single simulation run.

        Args:
            stl_file_path (Path): The path to the .stl file to evaluate.

        Returns:
            float: A stability score. Typically 1.0 for a stable model and
                   0.0 for an unstable one.
        """
        if not stl_file_path.exists():
            logger.error(f"STL file not found at {stl_file_path}")
            return 0.0
            
        try:
            self._setup_simulation()
            if self.client_id < 0: # Setup failed
                return 0.0
            
            model_id = self._load_model(stl_file_path)
            is_stable = self._run_stability_test(model_id)
            return 1.0 if is_stable else 0.0
        except p.error as e:
            # Catch PyBullet-specific errors, which often occur with bad meshes
            logger.error(f"PyBullet simulation failed for {stl_file_path.name}: {e}")
            return 0.0
        except Exception as e:
            # Catch any other unexpected errors during simulation
            logger.error(f"An unexpected error occurred during physics simulation for {stl_file_path.name}: {e}")
            return 0.0
        finally:
            self._cleanup_simulation()