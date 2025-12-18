"""Fluid Simulation using Laplacian Eigenfunctions"""

from dataclasses import dataclass
from typing import List, Optional
from jaxtyping import Array, Float64, Int32, Int64
from jax import numpy as jnp, vmap
import jax

# 1. precomputation of the wave number for the 3D domain (k1, k2, k3) 
# 2. precomputation of the eigenvalues (lambda_k) 
# 3. precomputation of the velocity basis field 
# 4. precomputation of the matrix Ck using those things

class EigenFluid():
    """Fluid Simulation using Laplacial Eigenfunctions"""
    # Dynamic components
    basis_coef: Float64[Array, "N"] # Velocity/Vorticity basis coefficients
    force_coef: Float64[Array, "N"] # Projected external force coefficients (f_k)
    velocity_field: Float64[Array, "mx my mz 3"] # Current velocity field

    # Static (or precomputed) components
    mx: Int32 # Velocity box size
    my: Int32
    mz: Int32 
    dt: Float64  # The fixed time step used for integration
    viscosity: Float64 # The viscosity parameter of the fluid
    domain_geometry: Array # The dimension
    N: Int64  # Basis Dimension
   
    vector_wave_numbers: Int32[Array, "N 4"] 
    eigenvalues: Float64[Array, "N"]  # Eigenvalues for viscosity and ordering modes

    velocity_basis_fields: Float64[Array, "N mx my mz 3"]  # Φ_k velocity basis
    Ck: Float64[Array, "N N N"] # Structure coefficients

    # Precomputation 

    # 1)
    def precompute_wave_numbers(self) -> None:
        N = int(self.N)
        if N % 3 != 0:
            raise ValueError(f"3D basis expects N divisible by 3; got N={N}")

        K = N // 3  # number of unique k-triples
        m = int(round(K ** (1.0 / 3.0)))
        if m**3 != K:
            raise ValueError(f"Expected N = 3*m^3. Got N={N}, so K=N/3={K} is not a perfect cube.")

        ks: Int32[Array, "m"] = jnp.arange(1, m + 1, dtype=jnp.int32)
        k1, k2, k3 = jnp.meshgrid(ks, ks, ks, indexing="ij")

        k_triples: Int32[Array, "K 3"] = jnp.stack(
            [k1.reshape(-1), k2.reshape(-1), k3.reshape(-1)],
            axis=1,
        )

        # Sort by |k|^2 = k1^2 + k2^2 + k3^2
        lam_mag: Int64[Array, "K"] = (
            k_triples[:, 0].astype(jnp.int64) ** 2
            + k_triples[:, 1].astype(jnp.int64) ** 2
            + k_triples[:, 2].astype(jnp.int64) ** 2
        )
        order: Int32[Array, "K"] = jnp.argsort(lam_mag).astype(jnp.int32)
        k_triples = k_triples[order]

        # Expand each k-triple into three forms i=1,2,3
        k_rep: Int32[Array, "N 3"] = jnp.repeat(k_triples, 3, axis=0)
        i_rep: Int32[Array, "N 1"] = jnp.tile(jnp.arange(1, 4, dtype=jnp.int32), K)[:, None]

        self.vector_wave_numbers: Int32[Array, "N 4"] = jnp.concatenate([k_rep, i_rep], axis=1)

    # 3D box from the thesis:
    # DE WITT, T. 2010. Fluid Simulation in Bases of Laplacian Eigenfunctions. 
    # M.S. thesis, University of Toronto, Toronto, ON, Canada.
    @staticmethod
    def analytic_velocity_basis(
        k_mode: Int32[Array, "4"],          # (k1,k2,k3,i) with i ∈ {1,2,3}
        x: Float64[Array, ""],
        y: Float64[Array, ""],
        z: Float64[Array, ""],
    ) -> Float64[Array, "3"]:
        k1 = k_mode[0].astype(jnp.float64)
        k2 = k_mode[1].astype(jnp.float64)
        k3 = k_mode[2].astype(jnp.float64)
        i  = k_mode[3].astype(jnp.int32)    # 1..3

        lam = k1**2 + k2**2 + k3**2
        inv_lam = 1.0 / lam

        s1, c1 = jnp.sin(k1 * x), jnp.cos(k1 * x)
        s2, c2 = jnp.sin(k2 * y), jnp.cos(k2 * y)
        s3, c3 = jnp.sin(k3 * z), jnp.cos(k3 * z)

        # Φ_{k,1}
        v1 = jnp.stack([
            inv_lam * (k2**2 + k3**2) * c1 * s2 * s3,
            inv_lam * (-k1 * k2)      * s1 * c2 * s3,
            inv_lam * (-k1 * k3)      * s1 * s2 * c3,
        ])

        # Φ_{k,2}
        v2 = jnp.stack([
            inv_lam * (k2 * k1)        * c1 * s2 * s3,
            inv_lam * (-(k1**2+k3**2)) * s1 * c2 * s3,
            inv_lam * (k2 * k3)        * s1 * s2 * c3,
        ])

        # Φ_{k,3}
        v3 = jnp.stack([
            inv_lam * (-(k3 * k1))     * c1 * s2 * s3,
            inv_lam * (-(k3 * k2))     * s1 * c2 * s3,
            inv_lam * (k1**2 + k2**2)  * s1 * s2 * c3,
        ])

        return jax.lax.switch(i - 1, [lambda: v1, lambda: v2, lambda: v3])

    def precompute_velocity_basis_fields(self):
        mx, my, mz  = self.mx, self.my, self.mz

        # Assuming a pi * pi * pi domain?
        xs = (jnp.arange(mx, dtype=jnp.float64) + 0.5) * (jnp.pi / mx)
        ys = (jnp.arange(my, dtype=jnp.float64) + 0.5) * (jnp.pi / my)
        zs = (jnp.arange(mz, dtype=jnp.float64) + 0.5) * (jnp.pi / mz)
        X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")  # (mx,my,mz)

        # Evaluate one mode on the whole grid (vectorize over grid points)
        eval_on_grid = jnp.vectorize(
            lambda km, x, y, z: EigenFluid.analytic_velocity_basis(km, x, y, z),
            signature="(4),(),(),()->(3)",
        )

        # vmap over modes
        result = jax.vmap(lambda km: eval_on_grid(km, X, Y, Z), in_axes=0)(self.vector_wave_numbers).astype(jnp.float64)

        self.velocity_basis_fields = result

    @staticmethod
    def curl_3d(U: Float64[Array, "mx my mz 3"], dx: float, dy: float, dz: float) -> Float64[Array, "mx my mz 3"]:
        ux, uy, uz = U[..., 0], U[..., 1], U[..., 2]

        duz_dy = jnp.gradient(uz, dy, axis=1)
        duy_dz = jnp.gradient(uy, dz, axis=2)

        dux_dz = jnp.gradient(ux, dz, axis=2)
        duz_dx = jnp.gradient(uz, dx, axis=0)

        duy_dx = jnp.gradient(uy, dx, axis=0)
        dux_dy = jnp.gradient(ux, dy, axis=1)

        wx = duz_dy - duy_dz
        wy = dux_dz - duz_dx
        wz = duy_dx - dux_dy
        
        return jnp.stack([wx, wy, wz], axis=-1)

    def precompute_Ck(self):
        Phi = self.velocity_basis_fields                      # (N,mx,my,mz,3)
        N = int(self.N)
        mx, my, mz = int(self.mx), int(self.my), int(self.mz)

        # grid spacing for [0, π]^3 sampled at cell centers
        dx = jnp.pi / mx
        dy = jnp.pi / my
        dz = jnp.pi / mz

        # Compute vorticity basis numerically: curl(Phi_k)
        # Curl each mode: vmap over N
        Phi_curl = jax.vmap(lambda U: EigenFluid.curl_3d(U, dx, dy, dz), in_axes=0)(Phi)  # (N,mx,my,mz,3)

        # Flatten spatial dims -> M points
        Phi_f      = Phi.reshape(N, -1, 3)       # (N,M,3)
        Phi_curl_f = Phi_curl.reshape(N, -1, 3)  # (N,M,3)
        lam = self.eigenvalues                   # (N,)

        # Optional integration weight (approx ∫ ≈ sum * dV)
        dV = dx * dy * dz

        # Allocate output
        Ck = jnp.zeros((N, N, N), dtype=Phi.dtype)

        def body(j, Ck_acc):
            # cross(Phi_i, Phi_j) for all i, all points -> (N,M,3)
            cross_ij = jnp.cross(Phi_f, Phi_f[j][None, :, :])   # broadcast over i

            # project onto all k using dot with curl(Phi_k):
            # out[k,i] = sum_m cross_ij[i,m,:] · Phi_curl_f[k,m,:]
            # einsum: (i m c, k m c) -> (k i)
            proj_ki = jnp.einsum("imc,kmc->ki", cross_ij, Phi_curl_f)

            # scale by λ_j and volume element
            slice_kij = (lam[j] * dV) * proj_ki   # (k,i)

            # write into Ck[:, :, j]
            Ck_acc = Ck_acc.at[:, :, j].set(slice_kij)
            return Ck_acc

        Ck = jax.lax.fori_loop(0, N, body, Ck)

        def threshold_zero(Ck: Float64[Array, "N N N"], eps: float = 1e-6) -> Float64[Array, "N N N"]:
            return jnp.where(jnp.abs(Ck) > eps, Ck, 0.0)

        self.Ck = threshold_zero(Ck, eps=1e-6)


    # 2. 
    def precompute_eigenvalues(self):
        """
        Laplacian eigenvalues for our domain:
            λ_k = (k1^2 + k2^2 + k3^2)
        """
        k = self.vector_wave_numbers
        lam_mag: Int64[Array, "N"] = (
            k[:, 0].astype(jnp.int64) ** 2
            + k[:, 1].astype(jnp.int64) ** 2
            + k[:, 2].astype(jnp.int64) ** 2
        )
        result = lam_mag.astype(jnp.float64)

        self.eigenvalues = result


    def expand_basis(self) -> Float64[Array, "x y z 3"]:
        u = jnp.einsum("n, nxyzv -> xyzv", self.basis_coef, self.velocity_basis_fields)
        return u

    def step(self):
        """
        Fluid simulator (based on Algorithm 1 in the paper). This basically 
        advances the simulation.
        """
        
        w = self.basis_coef = jnp.zeros((self.N,), dtype=jnp.float32) 
        N = self.N
        
        # Store kinetic energy of the velocity field
        e1 = jnp.sum(w**2)
        
        # Matrix vector product
        # Transpose of `w` happens using this `subscript` argument
        w_dot = jnp.einsum('i, kij, j -> k', w, self.Ck, w)

        # Explicit Euler integration
        w_adv = w + w_dot * self.dt
        
        # Calculate energy after time step
        e2 = jnp.sum(w_adv**2) 
        
        # Renormalize energy
        w_renorm = jnp.where(
            e2 > 1e-5,  # From Java implementation
            w_adv * jnp.sqrt(e1 / e2), 
            w_adv
        )
        
        # Dissipate energy for viscosity
        viscosity_factors = jnp.exp(-1.0 * self.eigenvalues * self.viscosity * self.dt)
        w_viscous = w_renorm * viscosity_factors
        
        # External Forces
        w_final = w_viscous + self.force_coef

        self.basis_coef = w_final 
        self.force_coef = jnp.zeros_like(self.force_coef)

        # Reconstruct velocity field
        self.velocity_field = self.expand_basis()