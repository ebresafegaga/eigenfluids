"""Fluid Simulation using Laplacian Eigenfunctions"""

from typing import List, Optional
from jaxtyping import Array, Float64, Int32, Int64
from jax import numpy as jnp, vmap
import jax

# approach to implementation
# 1. precomputation of the wave number for the 3D domain (k1, k2, k3) 
# 2. precomputation of the eigenvalues (lambda_k) 
# 3. precomputation of the velocity basis field 
# 4. precomputation of the matrix Ck using those things
# 5. implement the time step (should be easy)

def sample_trilinear_cellcenter(U, pos):
    """
    U: (mx,my,mz,3) or (mx,my,mz) sampled at cell centers in [0,pi]^3
    pos: (...,3) in [0,pi]^3
    returns: (...,3) or (...) sampled by trilinear interpolation
    """
    mx, my, mz = U.shape[0], U.shape[1], U.shape[2]
    L = jnp.pi

    # map x in [0,L] -> grid coordinate g in [ -0.5, mx-0.5 ]
    gx = (pos[..., 0] / L) * mx - 0.5
    gy = (pos[..., 1] / L) * my - 0.5
    gz = (pos[..., 2] / L) * mz - 0.5

    i0 = jnp.floor(gx).astype(jnp.int32)
    j0 = jnp.floor(gy).astype(jnp.int32)
    k0 = jnp.floor(gz).astype(jnp.int32)

    tx = gx - i0
    ty = gy - j0
    tz = gz - k0

    # clamp to valid cell-center range
    i0 = jnp.clip(i0, 0, mx - 2)
    j0 = jnp.clip(j0, 0, my - 2)
    k0 = jnp.clip(k0, 0, mz - 2)

    i1 = i0 + 1
    j1 = j0 + 1
    k1 = k0 + 1

    def at(ii, jj, kk):
        return U[ii, jj, kk]

    c000 = at(i0, j0, k0)
    c100 = at(i1, j0, k0)
    c010 = at(i0, j1, k0)
    c110 = at(i1, j1, k0)
    c001 = at(i0, j0, k1)
    c101 = at(i1, j0, k1)
    c011 = at(i0, j1, k1)
    c111 = at(i1, j1, k1)

    c00 = c000 * (1 - tx)[..., None] + c100 * tx[..., None] if c000.ndim == pos.ndim else c000*(1-tx)+c100*tx
    c10 = c010 * (1 - tx)[..., None] + c110 * tx[..., None] if c010.ndim == pos.ndim else c010*(1-tx)+c110*tx
    c01 = c001 * (1 - tx)[..., None] + c101 * tx[..., None] if c001.ndim == pos.ndim else c001*(1-tx)+c101*tx
    c11 = c011 * (1 - tx)[..., None] + c111 * tx[..., None] if c011.ndim == pos.ndim else c011*(1-tx)+c111*tx

    c0 = c00 * (1 - ty)[..., None] + c10 * ty[..., None] if c00.ndim == pos.ndim else c00*(1-ty)+c10*ty
    c1 = c01 * (1 - ty)[..., None] + c11 * ty[..., None] if c01.ndim == pos.ndim else c01*(1-ty)+c11*ty

    out = c0 * (1 - tz)[..., None] + c1 * tz[..., None] if c0.ndim == pos.ndim else c0*(1-tz)+c1*tz
    return out

def advect_particles(particles, U, dt, key=None, diffusion=0.01):
    # particles: (P,3)
    v1 = sample_trilinear_cellcenter(U, particles)                 # (P,3)
    mid = particles + 0.5 * dt * v1
    v2 = sample_trilinear_cellcenter(U, mid)
    newp = particles + dt * v2

    # Add random diffusion to prevent clustering along streamlines
    if key is not None:
        noise = jax.random.normal(key, particles.shape) * diffusion
        newp = newp + noise

    # keep in domain [0, pi]^3 (simple clamp)
    L = jnp.pi
    newp = jnp.clip(newp, 0.0, L)
    return newp

def advect_density(density, U, dt):
    mx, my, mz = density.shape
    L = jnp.pi

    xs = (jnp.arange(mx) + 0.5) * (L / mx)
    ys = (jnp.arange(my) + 0.5) * (L / my)
    zs = (jnp.arange(mz) + 0.5) * (L / mz)
    X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")
    pos = jnp.stack([X, Y, Z], axis=-1)  # (mx,my,mz,3)

    v = sample_trilinear_cellcenter(U, pos)              # (mx,my,mz,3)
    back = pos - dt * v
    back = jnp.clip(back, 0.0, L)

    # sample old density at backtraced points
    rho_new = sample_trilinear_cellcenter(density, back) # (mx,my,mz)
    return rho_new


class EigenFluid():
    """Fluid Simulation using Laplacial Eigenfunctions"""
    # Dynamic components
    basis_coef: Float64[Array, "N"] # Velocity/Vorticity basis coefficients
    force_coef: Float64[Array, "N"] # Projected external force coefficients (f_k)
    velocity_field: Float64[Array, "mx my mz 3"] # Current velocity field

    # Static (or precomputed) components
    mx: Int32 # the domain, this particular 
    my: Int32 # instance of eigenfluids is for a 3D box of pi * pi * pi
    mz: Int32 
    dt: Float64  # TTime step
    viscosity: Float64 # The viscosityof the fluid
    N: Int64  # The basis size (number of eigenfunctions 
              # -- more is better but requires more precomputation time )
   
    vector_wave_numbers: Int32[Array, "N 4"] 
    eigenvalues: Float64[Array, "N"]  # Eigenvalues for viscosity and ordering modes

    velocity_basis_fields: Float64[Array, "N mx my mz 3"]  # velocity basis fields
    Ck: Float64[Array, "N N N"] # Structure coefficients


    # Rendering stuff
    density: Float64[Array, "mx my mz"]                 # smoke density grid
    particles: Float64[Array, "P 3"]                    # particle positions in [0, pi]^3
    particle_rng_key: Array                             # random key for particle diffusion

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

        lam_mag: Int64[Array, "K"] = (
            k_triples[:, 0].astype(jnp.int64) ** 2
            + k_triples[:, 1].astype(jnp.int64) ** 2
            + k_triples[:, 2].astype(jnp.int64) ** 2
        )
        order: Int32[Array, "K"] = jnp.argsort(lam_mag).astype(jnp.int32)
        k_triples = k_triples[order]

        k_rep: Int32[Array, "N 3"] = jnp.repeat(k_triples, 3, axis=0)
        i_rep: Int32[Array, "N 1"] = jnp.tile(jnp.arange(1, 4, dtype=jnp.int32), K)[:, None]

        self.vector_wave_numbers: Int32[Array, "N 4"] = jnp.concatenate([k_rep, i_rep], axis=1)

    # 2. 
    # This function is dependent on calling precompute_wave_numbers first
    def precompute_eigenvalues(self):
        k = self.vector_wave_numbers
        lam_mag: Int64[Array, "N"] = (
            k[:, 0].astype(jnp.int64) ** 2
            + k[:, 1].astype(jnp.int64) ** 2
            + k[:, 2].astype(jnp.int64) ** 2
        )
        result = lam_mag.astype(jnp.float64)

        self.eigenvalues = result

    # 3D box from the thesis:
    # DE WITT, T. 2010. Fluid Simulation in Bases of Laplacian Eigenfunctions. 
    # M.S. thesis, University of Toronto, Toronto, ON, Canada.
    @staticmethod
    def closed_form_velocity_basis(
        wave_numbers: Int32[Array, "4"],
        x: Float64,
        y: Float64,
        z: Float64,
    ) -> Float64[Array, "3"]:
        k1 = wave_numbers[0].astype(jnp.float64)
        k2 = wave_numbers[1].astype(jnp.float64)
        k3 = wave_numbers[2].astype(jnp.float64)
        i  = wave_numbers[3]

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
            lambda km, x, y, z: EigenFluid.closed_form_velocity_basis(km, x, y, z),
            signature="(4),(),(),()->(3)",
        )

        # vmap over modes
        result = jax.vmap(lambda km: eval_on_grid(km, X, Y, Z), in_axes=0)(self.vector_wave_numbers).astype(jnp.float64)

        self.velocity_basis_fields = result

    # used to calculate the curl of a velocity field to get the vorticity field
    # based on the curl operator defined in "Fluid Simulation for Computer Graphics" by Richard Bridson
    @staticmethod
    def curl(
        veclocity_field: Float64[Array, "mx my mz 3"], 
        dx: float, 
        dy: float, 
        dz: float
    ) -> Float64[Array, "mx my mz 3"]:
        ux, uy, uz = veclocity_field[..., 0], veclocity_field[..., 1], veclocity_field[..., 2]

        duz_dy = jnp.gradient(uz, dy, axis=1)
        duy_dz = jnp.gradient(uy, dz, axis=2)

        dux_dz = jnp.gradient(ux, dz, axis=2)
        duz_dx = jnp.gradient(uz, dx, axis=0)

        duy_dx = jnp.gradient(uy, dx, axis=0)
        dux_dy = jnp.gradient(ux, dy, axis=1)

        wx = duz_dy - duy_dz # pyright: ignore[reportOperatorIssue]
        wy = dux_dz - duz_dx # pyright: ignore[reportOperatorIssue]
        wz = duy_dx - dux_dy # pyright: ignore[reportOperatorIssue]
        
        return jnp.stack([wx, wy, wz], axis=-1)

    # The thesis doesn't provide an closed form solution for Ck, so we do it the numerical way
    # Based on Algorithm 2 from the paper
    def precompute_Ck(self):
        Phi = self.velocity_basis_fields
        N = int(self.N)
        mx, my, mz = int(self.mx), int(self.my), int(self.mz)

        dx = jnp.pi / mx
        dy = jnp.pi / my
        dz = jnp.pi / mz

        # Compute vorticity basis numerically: curl(Phi)
        vorticity_basis_fields = jax.vmap(lambda U: EigenFluid.curl(U, dx, dy, dz), in_axes=0)(self.velocity_basis_fields)

        # Flatten spatial dims -> M points
        Phi_f = Phi.reshape(N, -1, 3)       # (N,M,3)
        Phi_curl_f = vorticity_basis_fields.reshape(N, -1, 3)  # (N,M,3)

        def body(j, Ck_acc):
            cross_ij = jnp.cross(Phi_curl_f[j][None, :, :], Phi_f[j]) 

            proj_ki = jnp.einsum("imc,kmc->ki", cross_ij, Phi_curl_f)
            slice_kij = (self.eigenvalues[j]) * proj_ki

            Ck_acc = Ck_acc.at[:, :, j].set(slice_kij)
            return Ck_acc

        Ck = jnp.zeros((N, N, N), dtype=Phi.dtype)
        Ck = jax.lax.fori_loop(0, N, body, Ck)

        # Make the matix sparse based on a threshold
        def threshold_zero(Ck: Float64[Array, "N N N"], eps: float = 1e-6) -> Float64[Array, "N N N"]:
            return jnp.where(jnp.abs(Ck) > eps, Ck, 0.0)

        self.Ck = Ck # threshold_zero(Ck)


    def expand_basis(self) -> Float64[Array, "x y z 3"]:
        u = jnp.einsum("n, nxyzv -> xyzv", self.basis_coef, self.velocity_basis_fields)
        return u

    def step(self):
        """
        Fluid simulator (based on Algorithm 1 in the paper). This basically 
        advances the simulation.
        """
        
        w = self.basis_coef
        N = self.N
        
        # Store kinetic energy of the velocity field
        e1 = jnp.sum(w**2)
        
        # Matrix vector product
        # Transpose of `w` happens using the `subscript` argument
        w_dot = jnp.einsum('i, kij, j -> k', w, self.Ck, w)

        # Explicit Euler integration
        w_adv = w + w_dot * self.dt
        
        # Calculate energy after time step
        e2 = jnp.sum(w_adv**2) 
        
        # Renormalize energy
        # From Java implementation https://www.dgp.toronto.edu/~tyler/fluids/
        w_renorm = jnp.where(
            e2 > 1e-5, 
            w_adv * jnp.sqrt(e1 / e2), 
            w_adv
        )
        
        # Dissipate energy for viscosity
        viscosity_factors = jnp.exp(-1.0 * self.eigenvalues * self.viscosity * self.dt)
        w_viscous = w_renorm * viscosity_factors
        
        # External Forces
        w_final = w_viscous + self.force_coef

        self.basis_coef = w_final 
        # From the Java impl
        # I'm not sure why we need to set the force to zero...
        self.force_coef = jnp.zeros_like(self.force_coef)

        # Reconstruct velocity field
        self.velocity_field = self.expand_basis()

        # Rendering
        self.density = advect_density(self.density, self.velocity_field, self.dt)

        # Update particles with diffusion
        self.particle_rng_key, subkey = jax.random.split(self.particle_rng_key)
        self.particles = advect_particles(self.particles, self.velocity_field, self.dt,
                                         key=subkey, diffusion=0.008)