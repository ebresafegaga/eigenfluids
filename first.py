"""
Smoke density visualization with transparent box boundaries using Polyscope
"""

import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import jax.numpy as jnp
import jax
from eigenfluids import EigenFluid

def create_box_wireframe():
    """
    Create wireframe edges for a π×π×π box.
    Returns vertices and edges for the box outline.
    """
    # 8 vertices of the box
    vertices = np.array([
        [0, 0, 0], [np.pi, 0, 0], [np.pi, np.pi, 0], [0, np.pi, 0],  # bottom
        [0, 0, np.pi], [np.pi, 0, np.pi], [np.pi, np.pi, np.pi], [0, np.pi, np.pi]  # top
    ])
    
    # 12 edges of the box
    edges = np.array([
        # Bottom face
        [0, 1], [1, 2], [2, 3], [3, 0],
        # Top face
        [4, 5], [5, 6], [6, 7], [7, 4],
        # Vertical edges
        [0, 4], [1, 5], [2, 6], [3, 7]
    ])
    
    return vertices, edges


def register_smoke_volume(sim, name="smoke_density"):
    """
    Register the smoke density as a volume grid in polyscope.
    """
    # Get density field as numpy array
    density = np.array(sim.density)
    mx, my, mz = density.shape
    
    # Register volume grid with proper bounds
    ps.register_volume_grid(name, (mx, my, mz),
                           bound_low=(0., 0., 0.),
                           bound_high=(np.pi, np.pi, np.pi))

    # Add density as scalar quantity
    vol_grid = ps.get_volume_grid(name)
    vol_grid.add_scalar_quantity("density", density,
                                  enabled=True,
                                  vminmax=(0.0, 1.0),
                                  cmap='viridis')
    
    return vol_grid


def register_transparent_box():
    """
    Register transparent box boundaries.
    """
    vertices, edges = create_box_wireframe()
    
    # Register as curve network for wireframe
    box_network = ps.register_curve_network("box_boundary", vertices, edges)
    box_network.set_color([0.8, 0.8, 0.8])  # Light gray
    box_network.set_radius(0.01, relative=False)  # Thin lines

    # Also create transparent faces for the box
    # Define the 6 faces of the box (as triangles)
    faces = np.array([
        # Bottom (z=0)
        [0, 1, 2], [0, 2, 3],
        # Top (z=π)
        [4, 6, 5], [4, 7, 6],
        # Front (y=0)
        [0, 4, 5], [0, 5, 1],
        # Back (y=π)
        [3, 2, 6], [3, 6, 7],
        # Left (x=0)
        [0, 3, 7], [0, 7, 4],
        # Right (x=π)
        [1, 5, 6], [1, 6, 2]
    ])

    # Register transparent surface mesh
    box_surface = ps.register_surface_mesh("box_surface", vertices, faces)
    box_surface.set_transparency(0.1)  # Very transparent
    box_surface.set_color([0.9, 0.9, 1.0])  # Slight blue tint
    box_surface.set_smooth_shade(True)
    box_surface.set_edge_width(0.0)  # No edges on surface

    return box_network, box_surface


def attract_particles_to_point(particles, attraction_point=None, strength=0.2):
    """
    Attract particles towards a specific point by shrinking and offsetting.
    Similar to the Java attract_particles() method.

    Args:
        particles: (P, 3) array of particle positions
        attraction_point: (3,) point to attract to, default is lower-left area
        strength: how much to shrink (0.2 means shrink to 20% of size)
    """
    if attraction_point is None:
        # Default: attract to lower-left region like the Java code
        attraction_point = jnp.array([np.pi * 0.3, np.pi * 0.3, np.pi * 0.5])

    # Shrink particles towards origin, then offset to attraction point
    particles_attracted = particles * strength + attraction_point

    # Keep in domain
    particles_attracted = jnp.clip(particles_attracted, 0.0, jnp.pi)

    return particles_attracted


def visualize_smoke_simulation():
    """
    Main visualization function for smoke simulation.
    """
    # --- Configuration ---
    N_MODES_CUBED = 24 # 24, 81, 192, 375
    RES = 16            # Grid resolution
    
    # Initialize simulation
    print("Initializing simulation...")
    sim = EigenFluid()
    sim.N = N_MODES_CUBED
    sim.mx = sim.my = sim.mz = RES
    sim.dt = 0.01
    sim.viscosity = 0.005
    
    # --- Precomputations ---
    print("Precomputing basis and structure coefficients...")
    sim.precompute_wave_numbers()
    sim.precompute_eigenvalues()
    sim.precompute_velocity_basis_fields()
    sim.precompute_Ck()
    print("Precomputation complete.")
    
    # --- State Initialization ---
    # Initialize with more energy for dynamic smoke movement
    key = jax.random.PRNGKey(42)
    sim.basis_coef = jax.random.normal(key, (sim.N,)) * 0.4  # More initial energy
    sim.force_coef = jnp.zeros((sim.N,))

    # Initialize particle RNG key
    key, particle_key = jax.random.split(key)
    sim.particle_rng_key = particle_key
    
    # Initialize velocity field
    sim.velocity_field = sim.expand_basis()
    
    # Initialize Density - multiple thick smoke regions
    x = np.linspace(0, np.pi, RES)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    # Create multiple smoke sources with smooth falloff
    density = np.zeros((RES, RES, RES))

    # Large central smoke ball with gradual falloff
    dist1 = np.sqrt((X-np.pi/2)**2 + (Y-np.pi/2)**2 + (Z-np.pi/2)**2)
    density += np.exp(-dist1**2 / 0.3)  # Smoother, larger smoke cloud

    # Bottom-left smoke region
    dist2 = np.sqrt((X-np.pi/4)**2 + (Y-np.pi/4)**2 + (Z-np.pi/4)**2)
    density += np.exp(-dist2**2 / 0.2) * 0.9

    # Top-right smoke region
    dist3 = np.sqrt((X-3*np.pi/4)**2 + (Y-3*np.pi/4)**2 + (Z-3*np.pi/4)**2)
    density += np.exp(-dist3**2 / 0.25) * 0.85

    # Mid-level smoke layer (horizontal band)
    dist4 = np.abs(Z - np.pi/2)
    density += np.exp(-dist4**2 / 0.15) * 0.5

    # Normalize
    density = np.clip(density, 0, 1)
    sim.density = jnp.array(density)
    
    # Initialize Particles (optional, for additional visualization)
    # Sample more particles concentrated in high-density regions for better smoke visualization
    num_particles = 200000  # Much more particles for thick smoke

    # Sample particles weighted by density for more realistic smoke
    density_flat = density.flatten()
    density_prob = density_flat / (density_flat.sum() + 1e-10)

    # Sample grid cell indices weighted by density
    num_cells = RES * RES * RES
    particle_key, key = jax.random.split(key)
    cell_indices = jax.random.choice(particle_key, num_cells, shape=(num_particles,), p=density_prob)

    # Convert cell indices to 3D positions with random offsets within cells
    iz = cell_indices // (RES * RES)
    iy = (cell_indices % (RES * RES)) // RES
    ix = cell_indices % RES

    # Add random offsets within each cell
    offset_key, key = jax.random.split(key)
    offsets = jax.random.uniform(offset_key, (num_particles, 3)) * (np.pi / RES)

    particle_positions = jnp.stack([
        (ix + 0.5) * (np.pi / RES),
        (iy + 0.5) * (np.pi / RES),
        (iz + 0.5) * (np.pi / RES)
    ], axis=-1) + offsets

    sim.particles = particle_positions

    # --- Polyscope Setup ---
    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("z_up")
    ps.set_background_color([0.1, 0.1, 0.15])  # Dark background for better smoke visibility

    # Register box boundaries
    print("Creating box boundaries...")
    box_wireframe, box_surface = register_transparent_box()

    # Register initial smoke density
    print("Registering smoke volume...")
    vol_grid = register_smoke_volume(sim)

    # Optional: Register particles with thick smoke appearance
    particle_cloud = ps.register_point_cloud("particles", np.array(sim.particles))
    particle_cloud.set_radius(0.004, relative=False)  # Larger particles for thick smoke
    particle_cloud.set_color([0.85, 0.85, 0.9])  # Bright light gray/white for dense smoke
    particle_cloud.set_enabled(True)  # Start with particles enabled
    
    # Simulation state
    sim_state = {
        'running': False,
        'step_count': 0,
        'show_particles': False,
        'add_force': False,
        'force_strength': 0.1
    }
    
    def callback():
        """GUI callback for interactive controls."""
        # Simulation controls
        changed, sim_state['running'] = psim.Checkbox("Run Simulation", sim_state['running'])
        
        if psim.Button("Step Once"):
            sim.step()
            sim_state['step_count'] += 1

            # Update visualization
            vol_grid.add_scalar_quantity("density", np.array(sim.density),
                                        enabled=True, vminmax=(0.0, 1.0), cmap='viridis')
            particle_cloud.update_point_positions(np.array(sim.particles))
        
        if psim.Button("Reset Density"):
            # Reset to initial smoke configuration
            dist = np.sqrt((X-np.pi/2)**2 + (Y-np.pi/2)**2 + (Z-np.pi/2)**2)
            sim.density = jnp.array(np.where(dist < 0.5, 1.0, 0.0))
            vol_grid.add_scalar_quantity("density", np.array(sim.density),
                                        enabled=True, vminmax=(0.0, 1.0), cmap='viridis')

        if psim.Button("Attract Particles"):
            # Bunch up particles to watch them advect
            sim.particles = attract_particles_to_point(sim.particles, strength=0.2)
            particle_cloud.update_point_positions(np.array(sim.particles))

        # Display controls
        changed, sim_state['show_particles'] = psim.Checkbox("Show Particles", 
                                                             sim_state['show_particles'])
        if changed:
            particle_cloud.set_enabled(sim_state['show_particles'])
        
        # Force controls
        changed, sim_state['add_force'] = psim.Checkbox("Add Random Forces", 
                                                         sim_state['add_force'])
        changed, sim_state['force_strength'] = psim.SliderFloat("Force Strength", 
                                                                sim_state['force_strength'],
                                                                0.0, 0.5)
        
        # Info display
        psim.Text(f"Step: {sim_state['step_count']}")
        psim.Text(f"Total Energy: {np.sum(sim.basis_coef**2):.4f}")
        psim.Text(f"Max Density: {np.max(sim.density):.4f}")
        
        # Run continuous simulation
        if sim_state['running']:
            # Add forces if enabled
            if sim_state['add_force'] and sim_state['step_count'] % 20 == 0:
                key = jax.random.PRNGKey(sim_state['step_count'])
                random_force = jax.random.normal(key, (sim.N,)) * sim_state['force_strength']
                sim.force_coef = sim.force_coef + random_force
            
            # Step simulation
            sim.step()
            sim_state['step_count'] += 1
            
            # Update visualization every frame
            if sim_state['step_count'] % 1 == 0:  # Update every step
                vol_grid.add_scalar_quantity("density", np.array(sim.density),
                                            enabled=True, vminmax=(0.0, 1.0), cmap='viridis')
                if sim_state['show_particles']:
                    particle_cloud.update_point_positions(np.array(sim.particles))
    
    # Set callback and show
    ps.set_user_callback(callback)
    
    ps.show()


if __name__ == "__main__":
    visualize_smoke_simulation()