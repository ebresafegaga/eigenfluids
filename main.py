import polyscope as ps
import numpy as np

from eigenfluids import *

def register_velocity_basis_arrows(sim, n: int, *, stride: int = 2, name: str = "Phi"):
    """
    Visualize sim.velocity_basis_fields[n] as arrow glyphs on a downsampled grid.
    stride: use 2,3,4,... to reduce clutter.
    """
    # Pull field (mx,my,mz,3) -> numpy
    U = np.array(sim.velocity_basis_fields[n])  # (mx,my,mz,3)
    mx, my, mz, _ = U.shape

    # Downsample
    U = U[::stride, ::stride, ::stride, :]
    mx2, my2, mz2, _ = U.shape

    # Build matching grid point positions in [0, pi]^3 (cell centers)
    xs = (np.arange(mx2) + 0.5) * (np.pi / mx2)
    ys = (np.arange(my2) + 0.5) * (np.pi / my2)
    zs = (np.arange(mz2) + 0.5) * (np.pi / mz2)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    P = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)   # (num_points, 3)
    V = U.reshape(-1, 3)                               # (num_points, 3)

    pc = ps.register_point_cloud(f"{name}[{n}]", P)
    pc.add_vector_quantity("velocity", V, enabled=True)

# Initialize polyscope
ps.init()

sim = EigenFluid()
sim.N = 24
sim.mx = sim.my = sim.mz = 45
sim.force_coef = 1.0
sim.dt = 0.1
sim.viscosity = 1e-3
sim.precompute_wave_numbers()
sim.precompute_eigenvalues()
sim.precompute_velocity_basis_fields()
sim.precompute_Ck()

print(sim.Ck)

# def sparsity_report(Ck, eps=1e-10):
#     import jax.numpy as jnp
#     total = Ck.size
#     nnz = jnp.sum(jnp.abs(Ck) > eps)
#     return float(nnz / total), int(nnz), int(total)

# def sparsity_sweep(Ck):
#     for eps in [1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8, 1e-9]:
#         nnz = jnp.sum(jnp.abs(Ck) > eps)
#         frac = nnz / Ck.size
#         print(f"eps={eps:>8g}  nnz_frac={float(frac):.4f}  nnz={int(nnz)}")

# sparsity_sweep(sim.Ck)

# from jax.experimental.sparse import BCOO
# def sparsify_Ck(Ck, eps=1e-6):
#     mask = jnp.abs(Ck) > eps
#     idx = jnp.argwhere(mask)   # (nnz, 3)
#     data = Ck[mask]            # (nnz,)
#     return BCOO((data, idx), shape=Ck.shape)

# s = sparsify_Ck(sim.Ck, eps=1e-6)
# print("nnz:", s.nse)

# def wdot_from_Ck_dense(Ck, w):
#     return jnp.einsum("i,kij,j->k", w, Ck, w)

# def energy_drift(Ck, w):
#     wdot = wdot_from_Ck_dense(Ck, w)
#     return jnp.dot(w, wdot)  # should be ~0

# w = jnp.ones((int(sim.N),), dtype=jnp.float32) * 0.1
# print("drift (dense):", float(energy_drift(sim.Ck, w)))

# Ck_thr = jnp.where(jnp.abs(sim.Ck) > 1e-6, sim.Ck, 0.0)
# print("drift (thr):  ", float(energy_drift(Ck_thr, w)))

# frac, nnz, total = sparsity_report(sim.Ck, eps=1e-9)
# print("nnz fraction:", frac, "nnz:", nnz, "total:", total)

register_velocity_basis_arrows(sim, n=24, stride=2, name="Phi")

# View the smoke particles and box in the 3D UI
ps.show()