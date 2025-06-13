# 3d_lspo/oracles/physics_verifier.py

"""
Interfaces with the PyBullet physics engine to perform stability tests on
generated 3D models.

This module provides the core functionality for loading a 3D model (from an STL file)
into a physics simulation, applying forces or loads, and evaluating its physical
integrity and stability. This serves as a key "oracle" in the reinforcement
learning loop, providing feedback on the functional viability of a generated design.
"""

import time
import pybullet as p
import pybullet_data
from pathlib import Path
from typing import Union, Dict, Any, Tuple
import numpy as np


class PhysicsSimulator:
    """
    A wrapper for a PyBullet simulation environment.

    This class handles the lifecycle of a physics simulation, including connecting
    to the physics server, setting up the scene (e.g., gravity, ground plane),
    loading objects, running the simulation, and cleaning up resources. It is
    designed to be used as a context manager.
    """

    def __init__(self, use_gui: bool = False):
        """
        Initializes the physics simulator and connects to the PyBullet server.

        Args:
            use_gui (bool): If True, connects to a graphical user interface for
                            debugging. If False, runs in headless mode (p.DIRECT),
                            which is faster for training.
        """
        self.mode = p.GUI if use_gui else p.DIRECT
        self.client_id = -1
        self.use_gui = use_gui

    def __enter__(self) -> 'PhysicsSimulator':
        """
        Enters the runtime context for the simulator.

        Connects to the server and sets up the basic scene.

        Returns:
            PhysicsSimulator: The current instance of the simulator.
        """
        try:
            self.client_id = p.connect(self.mode)
            if self.client_id < 0:
                raise ConnectionError("Failed to connect to PyBullet.")
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            self._setup_scene()
        except Exception as e:
            # If setup fails, ensure we disconnect before re-raising
            self.close()
            raise ConnectionError(f"PyBullet initialization failed: {e}") from e
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exits the runtime context, ensuring the connection is closed.
        """
        self.close()

    def _setup_scene(self, gravity: float = -9.81) -> None:
        """
        Sets up the basic simulation scene.

        This includes setting the gravitational force and creating a static
        ground plane.

        Args:
            gravity (float): The gravitational acceleration in the Z-axis.
        """
        p.setGravity(0, 0, gravity, physicsClientId=self.client_id)
        p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True, physicsClientId=self.client_id)

    def load_model(
        self,
        stl_file_path: Union[str, Path],
        base_position: Tuple[float, float, float],
        mass: float,
        scale: float = 1.0
    ) -> int:
        """
        Loads a model from an STL file into the simulation.

        Args:
            stl_file_path (Union[str, Path]): Path to the .stl file of the model.
            base_position (Tuple[float, float, float]): The initial (x, y, z)
                position to place the model.
            mass (float): The mass of the object in kg. If set to 0, the object is static.
            scale (float): The scaling factor to apply to the model upon loading.
                         Assumes STL units are meters; use 0.001 if they are mm.

        Returns:
            int: The unique ID of the loaded body in the simulation.
        
        Raises:
            ValueError: If the STL file cannot be loaded by PyBullet.
        """
        stl_path_str = str(stl_file_path)
        try:
            # Create a collision shape from the mesh.
            collision_shape_id = p.createCollisionShape(
                shapeType=p.GEOM_MESH,
                fileName=stl_path_str,
                meshScale=[scale, scale, scale],
                physicsClientId=self.client_id
            )
            if collision_shape_id < 0:
                raise ValueError(f"PyBullet failed to create collision shape for {stl_path_str}")

            # Create the dynamic object using the collision shape.
            body_id = p.createMultiBody(
                baseMass=mass,
                baseCollisionShapeIndex=collision_shape_id,
                basePosition=base_position,
                physicsClientId=self.client_id
            )
            return body_id
        except Exception as e:
            # PyBullet can throw generic pybullet.error for loading issues
            raise ValueError(f"Failed to load STL model '{stl_path_str}': {e}") from e

    def run_simulation(self, steps: int) -> None:
        """
        Steps the simulation forward for a given number of iterations.

        Args:
            steps (int): The number of simulation steps to perform.
                         (PyBullet default is 240 Hz, so 240 steps = 1 second).
        """
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self.client_id)
            if self.use_gui:
                time.sleep(1./240.) # Slow down for visualization

    def check_stability(
        self,
        body_id: int,
        stability_threshold: float = 0.95,
        movement_threshold: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Checks the stability of a given body after the simulation has run.

        Args:
            body_id (int): The unique ID of the body to check.
            stability_threshold (float): The minimum z-component of the object's up-vector
                                         to be considered 'not tipped over'. Close to 1 is upright.
            movement_threshold (float): The initial z-value threshold below which an item is considered
                                        to have fallen through the world.

        Returns:
            Dict[str, Any]: A dictionary containing stability metrics.
        """
        try:
            pos, orn = p.getBasePositionAndOrientation(body_id, physicsClientId=self.client_id)
        except p.error:
            # The body ID might be invalid if it was removed or never created.
            return {'is_stable': False, 'tipped_over': True, 'final_pos': None, 'reason': 'Invalid body ID'}

        rot_matrix = p.getMatrixFromQuaternion(orn)
        up_vector = np.array([rot_matrix[2], rot_matrix[5], rot_matrix[8]])

        tipped_over = up_vector[2] < stability_threshold
        fell_through = pos[2] < movement_threshold

        is_stable = not tipped_over and not fell_through
        
        return {
            'is_stable': is_stable,
            'tipped_over': tipped_over,
            'fell_through': fell_through,
            'final_pos': pos,
            'final_z_up_component': up_vector[2],
        }

    def close(self) -> None:
        """
        Disconnects from the PyBullet physics server.
        """
        if self.client_id >= 0 and p.isConnected(self.client_id):
            p.disconnect(physicsClientId=self.client_id)
        self.client_id = -1


def verify_stability(
    stl_file_path: Union[str, Path],
    simulation_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Performs a complete stability verification test for a given STL model.

    This function is the main public entry point for the physics verifier. It
    instantiates a PhysicsSimulator, loads the target model and any specified
    test loads, runs the simulation, and returns a dictionary of results.

    Example simulation_config:
    {
        "use_gui": False,
        "model_scale": 0.001,      # Assumes STL is in mm, converts to meters
        "simulation_steps": 2400,  # e.g., 10 seconds at 240 Hz
        "model_mass": 0.1,         # Mass of the stand itself (in kg)
        "test_load": {             # Object to place on the model
            "type": "box",
            "dimensions": [0.07, 0.01, 0.15], # L, W, H of a phone (in meters)
            "mass": 0.2,                      # Mass of the phone (in kg)
            "offset_from_model": [0, 0.02, 0.05] # Where to place it, relative to model's CoM
        }
    }

    Args:
        stl_file_path (Union[str, Path]): The path to the STL file to be tested.
        simulation_config (Dict[str, Any]): A dictionary defining the parameters
            for the simulation test scenario.

    Returns:
        A result dictionary. On success:
        {'passed_stability_check': bool, 'details': {<stability_dict>}}
        On failure (e.g., failed to load STL):
        {'passed_stability_check': False, 'details': {'reason': str}}
    """
    if not Path(stl_file_path).exists():
        return {'passed_stability_check': False, 'details': {'reason': f"STL file not found: {stl_file_path}"}}

    try:
        with PhysicsSimulator(use_gui=simulation_config.get('use_gui', False)) as sim:
            
            # --- Load Primary Model ---
            model_id = sim.load_model(
                stl_file_path=stl_file_path,
                base_position=(0, 0, 0.5), # Start high and let it drop to settle
                mass=simulation_config.get('model_mass', 0.1),
                scale=simulation_config.get('model_scale', 1.0)
            )

            # Let the primary model settle on the ground
            sim.run_simulation(steps=480) # 2 seconds to settle

            model_settled_pos, _ = p.getBasePositionAndOrientation(model_id, physicsClientId=sim.client_id)

            # --- Load Test Load if specified ---
            if 'test_load' in simulation_config:
                load_config = simulation_config['test_load']
                load_offset = load_config.get('offset_from_model', [0, 0, 0])
                load_pos = (
                    model_settled_pos[0] + load_offset[0],
                    model_settled_pos[1] + load_offset[1],
                    model_settled_pos[2] + load_offset[2]
                )
                
                if load_config.get('type') == 'box':
                    dims = np.array(load_config.get('dimensions', [0.1, 0.1, 0.1]))
                    col_shape = p.createCollisionShape(
                        p.GEOM_BOX, halfExtents=dims / 2, physicsClientId=sim.client_id
                    )
                    p.createMultiBody(
                        baseMass=load_config.get('mass', 0.1),
                        baseCollisionShapeIndex=col_shape,
                        basePosition=load_pos,
                        physicsClientId=sim.client_id
                    )
                # Other load types could be added here (e.g., sphere, cylinder)
            
            # --- Run Main Simulation ---
            sim.run_simulation(steps=simulation_config.get('simulation_steps', 2400)) # Default 10 seconds

            # --- Check Stability ---
            stability_results = sim.check_stability(
                body_id=model_id,
                stability_threshold=simulation_config.get('stability_threshold', 0.95),
                movement_threshold=model_settled_pos[2] / 2 # Consider fallen if z is less than half its settled height
            )

            return {
                'passed_stability_check': stability_results.get('is_stable', False),
                'details': stability_results
            }

    except (ValueError, ConnectionError, p.error) as e:
        return {
            'passed_stability_check': False,
            'details': {'reason': f"Physics simulation failed: {e}"}
        }