"""Fluid Simulation using Laplacian Eigenfunctions"""

from dataclasses import dataclass
from typing import List
from jaxtyping import Array, Float64, Int32, Int64
from jax import numpy as jnp

@dataclass
class EigenFluid():
    # Dynamic components produced at each time step
    basis_coef: Float64[Array, "N"] # Velocity/Vorticity basis coefficients
    force_coef: Float64[Array, "N"] # Projected external force coefficients (f_k)
    # velocity_field: Float64

    # Statically known or precomputed
    dt: Float64  # The fixed time step used for integration
    viscosity: Float64 # The kinematic viscosity parameter of the fluid
    domain_geometry: Array # Geometry definition (e.g., bounding box dimensions)
    N: Int64  # Basis Dimension
    Ck:  Float64[Array, "N N N"] # Structure coefficients
    eigenvalues: Float64[Array, "N"]  # Eigenvalues for viscosity and ordering modes
    # velocity_basis_fields: Float64[Array, "N G G G Vec3"]

# Time integration
def step(e: EigenFluid) -> EigenFluid:
    """
    Integrates the basis coefficients (w) forward in time using an explicit Euler scheme 
    with energy renormalization, followed by viscosity and external force application.
    """
    
    # Current coefficient vector
    w = e.basis_coef
    N = e.N
    
    # --- 1. Store Initial Kinetic Energy ---
    # Kinetic energy E is calculated as the sum of squared coefficients (due to orthogonality) [3, 4].
    e1 = jnp.sum(w**2) 
    
    # --- 2. Calculate Non-linear Advection Term (ẇ = w^T * Ck * w) ---
    # The time derivative (ẇ) is computed component-wise using the precomputed Ck matrices [1, 5].
    
    # w_dot stores ẇ, the tangent vector representing the rate of change of coefficients
    # This calculation computes the non-linear part of the dynamics equation: ẇ_k = w^T Ck w
    
    # The batch dot product (jnp.einsum or vmap) is typically used for efficiency in JAX 
    # for this type of operation (w^T Ck w for all k).
    
    # This performs w[i] * Ck[k, i, j] * w[j] summed over i and j, resulting in a vector w_dot[k].
    w_dot = jnp.einsum('i, kij, j -> k', w, e.Ck, w)

    # --- 3. Explicit Euler Integration (Inviscid Step) ---
    # Perform an unconstrained timestep using the tangent vector [1, 6].
    w_adv = w + w_dot * e.dt
    
    # --- 4. Energy Renormalization ---
    # Calculate energy after the unconstrained step [1].
    e2 = jnp.sum(w_adv**2) 
    
    # Renormalize to preserve the kinetic energy (e1), projecting the state vector 
    # back onto the N-sphere manifold [6-8].
    
    # Using jnp.where for safe renormalization (avoiding issues if e2 is near zero)
    w_renorm = jnp.where(
        e2 > 1e-12, 
        w_adv * jnp.sqrt(e1 / e2), 
        w_adv
    )
    
    # --- 5. Viscosity Dissipation ---
    # Apply physical viscosity by decaying each coefficient exponentially [9, 10].
    # Decay factor is exp(ν * λ_k * Δt) [9, 11].
    
    viscosity_factors = jnp.exp(e.eigenvalues * e.viscosity * e.dt)
    w_viscous = w_renorm * viscosity_factors
    
    # --- 6. External Forces ---
    # External forces (f_k) are added as a linear term after viscosity [1, 2, 10].
    w_final = w_viscous + e.force_coef

    # --- 7. Return Updated State ---
    # Return a new EigenFluid object with the updated basis coefficients.
    return EigenFluid(
        # Dynamic
        basis_coef=w_final,
        force_coef=jnp.zeros_like(e.force_coef),
        # Static
        dt=e.dt,
        viscosity=e.viscosity,
        domain_geometry=e.domain_geometry,
        N=e.N,
        Ck=e.Ck,
        eigenvalues=e.eigenvalues,
    )