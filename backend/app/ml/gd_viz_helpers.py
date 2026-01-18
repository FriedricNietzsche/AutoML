"""
Gradient descent visualization event helpers.
Generates loss surface and GD path events per STAGE3_GRADIENT_DESCENT_EVENTS.md
"""
import numpy as np
from typing import Dict, Any, List
import math


def generate_loss_surface_spec(surface_kind: str = "bowl") -> Dict[str, Any]:
    """
    Generate a loss surface specification for 3D visualization.
    
    Args:
        surface_kind: Type of surface ("bowl", "multi_hill", "ripples", "saddle")
    
    Returns:
        Loss surface specification dict
    """
    return {
        "kind": surface_kind,
        "params": {},
        "domainHalf": 6.0,
        "zScale": 0.3
    }


def generate_gd_path(steps: int, convergence_rate: float = 0.1) -> List[Dict[str, float]]:
    """
    Generate a synthetic gradient descent path in normalized 2D space.
    
    Args:
        steps: Number of steps in the path
        convergence_rate: How quickly the path converges (higher = faster)
    
    Returns:
        List of normalized {x, y} points
    """
    # Start at a random point away from origin
    x, y = np.random.uniform(-0.8, 0.8, 2)
    
    path = []
    for step in range(steps):
        # Add current point
        path.append({"x": float(x), "y": float(y)})
        
        # Move towards origin with some noise
        dx = -x * convergence_rate + np.random.normal(0, 0.02)
        dy = -y * convergence_rate + np.random.normal(0, 0.02)
        
        x += dx
        y += dy
        
        # Clamp to stay in bounds
        x = np.clip(x, -0.95, 0.95)
        y = np.clip(y, -0.95, 0.95)
    
    return path


def generate_spiral_gd_path(steps: int) -> List[Dict[str, float]]:
    """
    Generate a spiral gradient descent path (looks cooler for demos).
    
    Args:
        steps: Number of steps
    
    Returns:
        List of normalized {x, y} points
    """
    path = []
    for step in range(steps):
        t = step / max(steps - 1, 1)
        r = 0.8 * (1 - t)  # Spiral inward
        theta = t * 4 * math.pi  # Multiple rotations
        
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        
        # Add noise
        x += np.random.normal(0, 0.02)
        y += np.random.normal(0, 0.02)
        
        path.append({"x": float(x), "y": float(y)})
    
    return path
