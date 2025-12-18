"""
Visualization of Laplacian Eigenfunction Basis Fields
Multiple techniques to match the paper/thesis visualizations
"""

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib import cm

def compute_vorticity(velocity_field):
    """
    Compute vorticity (curl) from velocity field.
    ω = ∇ × u
    """
    u, v, w = velocity_field[..., 0], velocity_field[..., 1], velocity_field[..., 2]
    mx, my, mz = u.shape
    
    # Use finite differences (assuming periodic boundaries)
    dx = dy = dz = np.pi / mx  # Assuming uniform spacing in π box
    
    # ∂w/∂y - ∂v/∂z
    omega_x = np.gradient(w, dy, axis=1) - np.gradient(v, dz, axis=2)
    
    # ∂u/∂z - ∂w/∂x  
    omega_y = np.gradient(u, dz, axis=2) - np.gradient(w, dx, axis=0)
    
    # ∂v/∂x - ∂u/∂y
    omega_z = np.gradient(v, dx, axis=0) - np.gradient(u, dy, axis=1)
    
    return np.stack([omega_x, omega_y, omega_z], axis=-1)


def visualize_basis_streamlines(sim, n, num_seeds=200):
    """
    Visualize basis function using streamlines (like in the paper).
    """
    # Get velocity field
    U = np.array(sim.velocity_basis_fields[n])
    mx, my, mz = U.shape[:3]
    
    # Create random seed points
    np.random.seed(42)
    seeds = np.random.rand(num_seeds, 3) * np.pi
    
    # Trace streamlines
    lines = []
    for seed in seeds:
        line = trace_streamline(U, seed, steps=50, dt=0.01)
        if len(line) > 5:  # Only keep meaningful streamlines
            lines.append(line)
    
    # Register streamlines with polyscope
    for i, line in enumerate(lines):
        if i > 50:  # Limit number of lines for performance
            break
        edges = np.column_stack([np.arange(len(line)-1), np.arange(1, len(line))])
        ps.register_curve_network(f"streamline_{i}", line, edges)
    
    return lines


def trace_streamline(velocity_field, seed, steps=100, dt=0.01):
    """
    Trace a single streamline using RK4 integration.
    """
    mx, my, mz = velocity_field.shape[:3]
    points = [seed]
    pos = seed.copy()
    
    for _ in range(steps):
        # Trilinear interpolation at current position
        vel = trilinear_interpolate(velocity_field, pos, mx, my, mz)
        
        if np.linalg.norm(vel) < 1e-6:
            break
            
        # RK4 integration
        k1 = vel
        k2 = trilinear_interpolate(velocity_field, pos + 0.5*dt*k1, mx, my, mz)
        k3 = trilinear_interpolate(velocity_field, pos + 0.5*dt*k2, mx, my, mz)
        k4 = trilinear_interpolate(velocity_field, pos + dt*k3, mx, my, mz)
        
        pos = pos + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
        
        # Periodic boundary conditions
        pos = np.mod(pos, np.pi)
        points.append(pos.copy())
        
    return np.array(points)


def trilinear_interpolate(field, pos, mx, my, mz):
    """
    Trilinear interpolation of field at position pos.
    """
    # Convert position to grid indices
    x, y, z = pos
    ix = (x / np.pi) * mx
    iy = (y / np.pi) * my
    iz = (z / np.pi) * mz
    
    # Get integer indices and fractions
    ix0 = int(np.floor(ix)) % mx
    iy0 = int(np.floor(iy)) % my
    iz0 = int(np.floor(iz)) % mz
    ix1 = (ix0 + 1) % mx
    iy1 = (iy0 + 1) % my
    iz1 = (iz0 + 1) % mz
    
    fx = ix - np.floor(ix)
    fy = iy - np.floor(iy)
    fz = iz - np.floor(iz)
    
    # Trilinear interpolation
    c000 = field[ix0, iy0, iz0]
    c001 = field[ix0, iy0, iz1]
    c010 = field[ix0, iy1, iz0]
    c011 = field[ix0, iy1, iz1]
    c100 = field[ix1, iy0, iz0]
    c101 = field[ix1, iy0, iz1]
    c110 = field[ix1, iy1, iz0]
    c111 = field[ix1, iy1, iz1]
    
    c00 = c000*(1-fx) + c100*fx
    c01 = c001*(1-fx) + c101*fx
    c10 = c010*(1-fx) + c110*fx
    c11 = c011*(1-fx) + c111*fx
    
    c0 = c00*(1-fy) + c10*fy
    c1 = c01*(1-fy) + c11*fy
    
    return c0*(1-fz) + c1*fz


def visualize_basis_isosurface(sim, n, level_fraction=0.3):
    """
    Visualize basis function using isosurfaces of vorticity magnitude.
    This matches many of the paper's visualizations.
    """
    # Get velocity field and compute vorticity
    U = np.array(sim.velocity_basis_fields[n])
    vorticity = compute_vorticity(U)
    vort_mag = np.linalg.norm(vorticity, axis=-1)
    
    # Smooth for cleaner isosurfaces
    vort_mag_smooth = gaussian_filter(vort_mag, sigma=1.0)
    
    # Create mesh grid
    mx, my, mz = U.shape[:3]
    xs = (np.arange(mx) + 0.5) * (np.pi / mx)
    ys = (np.arange(my) + 0.5) * (np.pi / my)
    zs = (np.arange(mz) + 0.5) * (np.pi / mz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    
    # Register as volume grid
    ps.register_volume_grid(f"basis_{n}_vorticity", (mx, my, mz),
                           bound_low=(0., 0., 0.),
                           bound_high=(np.pi, np.pi, np.pi))
    
    # Add scalar field
    vol_grid = ps.get_volume_grid(f"basis_{n}_vorticity")
    vol_grid.add_scalar_quantity("vorticity_magnitude", vort_mag_smooth, enabled=True)
    
    return vort_mag


def visualize_basis_slices(sim, n):
    """
    Visualize basis function using 2D slices (like Figure 5.1 in thesis).
    """
    U = np.array(sim.velocity_basis_fields[n])
    mx, my, mz = U.shape[:3]
    
    # Take slices at different z-planes
    z_slices = [mz//4, mz//2, 3*mz//4]
    
    fig, axes = plt.subplots(1, len(z_slices), figsize=(15, 5))
    
    for i, z_idx in enumerate(z_slices):
        # Get 2D slice
        u_slice = U[:, :, z_idx, 0]
        v_slice = U[:, :, z_idx, 1]
        speed = np.sqrt(u_slice**2 + v_slice**2)
        
        # Create coordinate grids for the slice
        xs = (np.arange(mx) + 0.5) * (np.pi / mx)
        ys = (np.arange(my) + 0.5) * (np.pi / my)
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        
        # Plot
        ax = axes[i]
        im = ax.contourf(X, Y, speed, levels=20, cmap='viridis')
        
        # Add streamlines
        skip = max(1, min(mx, my) // 15)
        ax.streamplot(xs[::skip], ys[::skip],
                     u_slice[::skip, ::skip], v_slice[::skip, ::skip],
                     color='white', density=1.0, linewidth=0.5)
        
        ax.set_title(f'z = {z_idx * np.pi / mz:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)
    
    plt.suptitle(f'Basis Function {n} - 2D Slices')
    plt.tight_layout()
    plt.savefig(f'basis_{n}_slices.png', dpi=150)
    plt.show()


def visualize_basis_glyphs_improved(sim, n, stride=3):
    """
    Improved glyph visualization with color-coded magnitude.
    """
    U = np.array(sim.velocity_basis_fields[n])
    mx, my, mz = U.shape[:3]
    
    # Downsample
    U_sub = U[::stride, ::stride, ::stride, :]
    mx2, my2, mz2 = U_sub.shape[:3]
    
    # Grid positions
    xs = (np.arange(mx2) * stride + 0.5) * (np.pi / mx)
    ys = (np.arange(my2) * stride + 0.5) * (np.pi / my)
    zs = (np.arange(mz2) * stride + 0.5) * (np.pi / mz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    
    P = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    V = U_sub.reshape(-1, 3)
    
    # Color by magnitude
    magnitudes = np.linalg.norm(V, axis=1)
    
    # Register with polyscope
    pc = ps.register_point_cloud(f"basis_{n}_points", P)
    vec = pc.add_vector_quantity("velocity", V, enabled=True, 
                                 radius=0.002, length=0.05)
    pc.add_scalar_quantity("magnitude", magnitudes, enabled=False)
    
    return pc


def visualize_all_modes(sim, max_modes=None):
    """
    Create a grid view of all basis functions.
    """
    if max_modes is None:
        max_modes = min(sim.N, 12)  # Limit for performance
    
    # Calculate grid layout
    cols = int(np.ceil(np.sqrt(max_modes)))
    rows = int(np.ceil(max_modes / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten() if max_modes > 1 else [axes]
    
    for n in range(max_modes):
        U = np.array(sim.velocity_basis_fields[n])
        mx, my, mz = U.shape[:3]
        
        # Take middle slice
        z_mid = mz // 2
        u_slice = U[:, :, z_mid, 0]
        v_slice = U[:, :, z_mid, 1]
        speed = np.sqrt(u_slice**2 + v_slice**2)
        
        ax = axes[n]
        im = ax.imshow(speed.T, cmap='viridis', origin='lower', 
                      extent=[0, np.pi, 0, np.pi])
        
        # Add wave number info
        k = sim.vector_wave_numbers[n]
        ax.set_title(f'n={n}: k=({k[0]},{k[1]},{k[2]}), i={k[3]}', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused subplots
    for n in range(max_modes, len(axes)):
        axes[n].set_visible(False)
    
    plt.suptitle('Basis Functions Overview (z=π/2 slice)')
    plt.tight_layout()
    plt.savefig('basis_overview.png', dpi=150)
    plt.show()


def main():
    # Initialize
    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("z_up")
    
    # Create simulation
    from eigenfluids import EigenFluid
    
    sim = EigenFluid()
    sim.N = 192  # Better for visualization
    sim.mx = sim.my = sim.mz = 32
    sim.precompute_wave_numbers()
    sim.precompute_eigenvalues()
    sim.precompute_velocity_basis_fields()
    
    print(f"Visualizing {sim.N} basis functions on {sim.mx}³ grid")
    
    # Choose visualization mode
    mode_to_visualize = 0  # Start with first mode
    
    # 1. Isosurface visualization (like the paper)
    print(f"Creating isosurface for mode {mode_to_visualize}...")
    visualize_basis_isosurface(sim, mode_to_visualize, level_fraction=0.2)
    
    # 2. Improved glyph visualization
    print(f"Creating glyph visualization for mode {mode_to_visualize}...")
    visualize_basis_glyphs_improved(sim, mode_to_visualize, stride=2)
    
    # 3. Create 2D slice plots
    print(f"Creating slice plots for mode {mode_to_visualize}...")
    visualize_basis_slices(sim, mode_to_visualize)
    
    # 4. Overview of all modes
    print("Creating overview of all modes...")
    visualize_all_modes(sim, max_modes=16)
    
    # 5. Optional: streamlines (slower)
    # print(f"Creating streamlines for mode {mode_to_visualize}...")
    # visualize_basis_streamlines(sim, mode_to_visualize, num_seeds=100)
    
    # Interactive mode selector
    def callback():
        changed, new_mode = psim.SliderInt("Mode", mode_to_visualize, 0, sim.N-1)
        if changed:
            # Clear previous visualizations
            ps.remove_all_structures()
            # Add new ones
            visualize_basis_isosurface(sim, new_mode)
            visualize_basis_glyphs_improved(sim, new_mode)
            return new_mode
        return mode_to_visualize
    
    # Show polyscope viewer
    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()