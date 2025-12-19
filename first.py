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


def visualize_smoke_simulation():
    """
    Main visualization function for smoke simulation.
    """
    # --- Configuration ---
    N_MODES_CUBED = 24  # 3 * 2^3
    RES = 16            # Grid resolution
    
    # Initialize simulation
    print("Initializing simulation...")
    sim = EigenFluid()
    sim.N = N_MODES_CUBED
    sim.mx = sim.my = sim.mz = RES 
    sim.dt = 0.01
    sim.viscosity = 0.01  # Lower viscosity for more movement
    
    # --- Precomputations ---
    print("Precomputing basis and structure coefficients...")
    sim.precompute_wave_numbers()
    sim.precompute_eigenvalues()
    sim.precompute_velocity_basis_fields()
    sim.precompute_Ck()
    print("Precomputation complete.")
    
    # --- State Initialization ---
    # Initialize with some energy
    key = jax.random.PRNGKey(42)
    sim.basis_coef = jax.random.normal(key, (sim.N,)) * 0.2
    sim.force_coef = jnp.zeros((sim.N,))
    
    # Initialize velocity field
    sim.velocity_field = sim.expand_basis()
    
    # Initialize Density - multiple smoke puffs
    x = np.linspace(0, np.pi, RES)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    
    # Create multiple smoke sources
    density = np.zeros((RES, RES, RES))
    
    # Central smoke ball
    dist1 = np.sqrt((X-np.pi/2)**2 + (Y-np.pi/2)**2 + (Z-np.pi/2)**2)
    density += np.where(dist1 < 0.5, 1.0, 0.0)
    
    # Corner smoke puff
    dist2 = np.sqrt((X-np.pi/4)**2 + (Y-np.pi/4)**2 + (Z-3*np.pi/4)**2)
    density += np.where(dist2 < 0.3, 0.8, 0.0)
    
    # Normalize
    density = np.clip(density, 0, 1)
    sim.density = jnp.array(density)
    
    # Initialize Particles (optional, for additional visualization)
    num_particles = 5000
    sim.particles = jax.random.uniform(key, (num_particles, 3)) * np.pi
    
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
    
    # Optional: Register particles
    particle_cloud = ps.register_point_cloud("particles", np.array(sim.particles))
    particle_cloud.set_radius(0.003, relative=False)
    particle_cloud.set_color([0.8, 0.8, 0.2])
    particle_cloud.set_enabled(False)  # Start with particles disabled
    
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
    
    print("\nControls:")
    print("- Use GUI to start/stop simulation")
    print("- Toggle particle visualization")
    print("- Add random forces to create turbulence")
    print("- Mouse to rotate view")
    
    ps.show()


if __name__ == "__main__":
    visualize_smoke_simulation()