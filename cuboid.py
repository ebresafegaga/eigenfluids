import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import jax.numpy as jnp
import jax
from eigenfluids_cuboid import EigenFluidCuboid

def create_box_cuboid(Lx, Ly, Lz):
    # vertices for cuboid Lx × Ly × Lz
    vertices = np.array([
        [0, 0, 0], [Lx, 0, 0], [Lx, Ly, 0], [0, Ly, 0],  # bottom
        [0, 0, Lz], [Lx, 0, Lz], [Lx, Ly, Lz], [0, Ly, Lz]  # top
    ])
    
    # edges
    edges = np.array([
        # Bottom face
        [0, 1], [1, 2], [2, 3], [3, 0],
        # Top face
        [4, 5], [5, 6], [6, 7], [7, 4],
        # Vertical edges
        [0, 4], [1, 5], [2, 6], [3, 7]
    ])
    
    return vertices, edges

def register_gas_volume(sim, name="smoke_density"):
    # Get density field as numpy array
    density = np.array(sim.density)
    mx, my, mz = density.shape

    ps.register_volume_grid(name, (mx, my, mz),
                           bound_low=(0., 0., 0.),
                           bound_high=(float(sim.Lx), float(sim.Ly), float(sim.Lz)))

    vol_grid = ps.get_volume_grid(name)
    vol_grid.add_scalar_quantity("density", density,
                                  enabled=True,
                                  vminmax=(0.0, 1.0),
                                  cmap='viridis')
    
    return vol_grid

def register_transparent_box(Lx, Ly, Lz):
    vertices, edges = create_box_cuboid(Lx, Ly, Lz)
    
    box_network = ps.register_curve_network("box_boundary", vertices, edges)
    box_network.set_color([0.8, 0.8, 0.8])  # Light gray
    box_network.set_radius(0.01, relative=False)  # Thin lines

    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1],
        [3, 2, 6], [3, 6, 7],
        [0, 3, 7], [0, 7, 4],
        [1, 5, 6], [1, 6, 2]
    ])

    # Register transparent surface mesh
    box_surface = ps.register_surface_mesh("box_surface", vertices, faces)
    box_surface.set_transparency(0.1)
    box_surface.set_color([0.9, 0.9, 1.0]) 
    box_surface.set_smooth_shade(True)
    box_surface.set_edge_width(0.0) 

    return box_network, box_surface

def visualize_gas_simulation():
    N_BASIS_DOMAIN = 192 # 24, 81, 192, 375
    RES = 16
    # the cuboid dims
    Lx, Ly, Lz = np.pi, np.pi, 2*np.pi 

    sim = EigenFluidCuboid()
    sim.N = N_BASIS_DOMAIN
    sim.mx = sim.my = sim.mz = RES
    sim.Lx, sim.Ly, sim.Lz = Lx, Ly, Lz
    sim.dt = 0.01
    sim.viscosity = 0.005
    
    print("Precomputing wave numbers")
    sim.precompute_wave_numbers()
    print("Precomputing wave eigenvalues")
    sim.precompute_eigenvalues()
    print("Precomputing wave velocity basis fields")
    sim.precompute_velocity_basis_fields()
    print("Precomputing structure coefficient matrix")
    sim.precompute_Ck()
    print("All precomputation complete")
    
    # initialize other stuff 

    key = jax.random.PRNGKey(42)
    sim.force_coef = jax.random.normal(key, (sim.N,)) * 0.4
    sim.basis_coef = jnp.zeros((sim.N,))

    key, particle_key = jax.random.split(key)
    sim.particle_rng_key = particle_key
    
    sim.velocity_field = sim.expand_basis()

    # Grid for tall box
    x = np.linspace(0, Lx, RES)
    y = np.linspace(0, Ly, RES)
    z = np.linspace(0, Lz, RES)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    density = np.zeros((RES, RES, RES))
    # Center smoke in the middle of the tall box
    dist1 = np.sqrt((X-Lx/2)**2 + (Y-Ly/2)**2 + (Z-Lz/2)**2)
    density += np.exp(-dist1**2 / 0.3)

    sim.density = jnp.array(density)
    
    num_particles = 200000  # Much more particles for thick smoke

    density_flat = density.flatten()
    density_prob = density_flat / (density_flat.sum() + 1e-10)

    num_cells = RES * RES * RES
    particle_key, key = jax.random.split(key)
    cell_indices = jax.random.choice(particle_key, num_cells, shape=(num_particles,), p=density_prob)

    iz = cell_indices // (RES * RES)
    iy = (cell_indices % (RES * RES)) // RES
    ix = cell_indices % RES

    offset_key, key = jax.random.split(key)
    offsets_x = jax.random.uniform(offset_key, (num_particles,)) * (Lx / RES)
    offset_key, key = jax.random.split(key)
    offsets_y = jax.random.uniform(offset_key, (num_particles,)) * (Ly / RES)
    offset_key, key = jax.random.split(key)
    offsets_z = jax.random.uniform(offset_key, (num_particles,)) * (Lz / RES)

    particle_positions = jnp.stack([
        (ix + 0.5) * (Lx / RES) + offsets_x,
        (iy + 0.5) * (Ly / RES) + offsets_y,
        (iz + 0.5) * (Lz / RES) + offsets_z
    ], axis=-1)

    sim.particles = particle_positions

 
    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("z_up")
    ps.set_background_color([0.1, 0.1, 0.15])

    register_transparent_box(Lx, Ly, Lz)
    vol_grid = register_gas_volume(sim)

    # Optional: Register particles with thick smoke appearance
    particle_cloud = ps.register_point_cloud("particles", np.array(sim.particles))
    particle_cloud.set_radius(0.005, relative=False)  # Larger particles for thick smoke
    particle_cloud.set_color([0.85, 0.85, 0.9])  # Bright light gray/white for dense smoke
    particle_cloud.set_enabled(True)  # Start with particles enabled
    
    sim_state = {
        'running': False,
        'step_count': 0,
        'show_particles': True,
        'add_force': False,
        'force_strength': 0.1
    }
    
    def callback():
        #  controls
        changed, sim_state['running'] = psim.Checkbox("Run Simulation", sim_state['running'])
        
        changed, sim_state['add_force'] = psim.Checkbox("Add Random Forces", 
                                                         sim_state['add_force'])
        changed, sim_state['force_strength'] = psim.SliderFloat("Force Strength", 
                                                                sim_state['force_strength'],
                                                                0.0, 0.5)
        
        # 
        psim.Text(f"Step: {sim_state['step_count']}")
        psim.Text(f"Total Energy: {np.sum(sim.basis_coef**2):.4f}")
        
        if sim_state['running']:
            # add random forces
            if sim_state['add_force'] and sim_state['step_count'] % 20 == 0:
                key = jax.random.PRNGKey(sim_state['step_count'])
                random_force = jax.random.normal(key, (sim.N,)) * sim_state['force_strength']
                sim.force_coef = sim.force_coef + random_force
            
            sim.step()
            sim_state['step_count'] += 1
            
            if sim_state['step_count'] % 1 == 0:
                vol_grid.add_scalar_quantity("density", np.array(sim.density),
                                            enabled=True, vminmax=(0.0, 1.0), cmap='viridis')
                if sim_state['show_particles']:
                    particle_cloud.update_point_positions(np.array(sim.particles))

    ps.set_user_callback(callback) 
    ps.show()


if __name__ == "__main__":
    visualize_gas_simulation()