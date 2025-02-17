# Copyright (c) 2024 Philipp Spiess
# All rights reserved.

import time
import os
import random
from io import BytesIO
from PIL import Image
import numpy as np
import pybullet as p
import pybullet_data
from settings import HEIGHT_SCALE, FRICTION_LOW, FRICTION_HIGH

# Constants
TEXTURE_SIZE = 256
NUM_ROWS = 128
NUM_COLUMNS = 128
GRAVITY = -9.81
NUM_ROW_BLOCK = 5  # Define this appropriately
FRICTION = 1.0  # Define this appropriately


# Decorator to time functions
def timeit(log=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            if log:
                print(f"Execution time of {func.__name__}: {end_time - start_time:.4f} seconds")
            return result

        return wrapper

    return decorator


# Ensure a folder exists
def create_folder_if_not_exists(folder_path: str) -> None:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


# Generate a random in-memory texture
def create_random_texture() -> bytes:
    random_texture = np.zeros((TEXTURE_SIZE, TEXTURE_SIZE, 3), dtype=np.uint8)
    random_texture[:, :, 1] = np.random.randint(0, 255, (TEXTURE_SIZE, TEXTURE_SIZE))
    image = Image.fromarray(random_texture)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# Create a random terrain with texture and friction
def create_random_terrain(p):
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Load the random texture
    texture_data = create_random_texture()
    texture_id = p.loadTexture(texture_data)

    # Generate height data
    height_data = np.random.uniform(-1, 1, size=(NUM_ROWS * NUM_COLUMNS)) * HEIGHT_SCALE
    terrain_shape = p.createCollisionShape(
        shapeType=p.GEOM_HEIGHTFIELD,
        meshScale=[1, 1, 1],
        heightfieldTextureScaling=3,
        heightfieldData=height_data,
        numHeightfieldRows=NUM_ROWS,
        numHeightfieldColumns=NUM_COLUMNS
    )
    terrain = p.createMultiBody(0, terrain_shape)
    p.changeVisualShape(terrain, -1, textureUniqueId=texture_id)
    p.setGravity(0, 0, GRAVITY)

    # Apply random friction values
    for _ in range(NUM_ROWS * NUM_COLUMNS // 10):  # Reduce iterations for efficiency
        friction_value = np.random.uniform(FRICTION_LOW, FRICTION_HIGH)
        p.changeDynamics(terrain, -1, lateralFriction=friction_value)

    return texture_id


# Add relief elements to the terrain
def add_relief(p):
    block_half_extents = [1, 1, 0.1]
    for i in range(-NUM_ROW_BLOCK, NUM_ROW_BLOCK + 1, 2):
        for j in range(-NUM_ROW_BLOCK, NUM_ROW_BLOCK + 1, 2):
            elevation = np.random.uniform(0.0, 0.2)
            blockPos = [i, j, elevation]
            blockOri = p.getQuaternionFromEuler([0, 0, 0])
            friction_block = random.uniform(0, FRICTION)

            blockId = p.createCollisionShape(p.GEOM_BOX, halfExtents=block_half_extents)
            visualShapeId = p.createVisualShape(p.GEOM_BOX, halfExtents=block_half_extents)

            multiBodyId = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=blockId,
                baseVisualShapeIndex=visualShapeId,
                basePosition=blockPos,
                baseOrientation=blockOri
            )
            p.changeDynamics(multiBodyId, -1, lateralFriction=friction_block)

    return p


# Log performance entries
def add_log_entry(log_message: str, file_name: str = 'performance.log') -> None:
    with open(file_name, 'a') as file:
        file.write(log_message + '\n')
