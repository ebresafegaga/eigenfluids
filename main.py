import polyscope as ps
import numpy as np
from particle import smoke_particles
from box import default_box_verts, default_box_faces

# Initialize polyscope
ps.init()

# Register smoke particles as a point cloud
smoke_cloud = ps.register_point_cloud("smoke particles", np.array(smoke_particles))

# Set the color to blue
smoke_cloud.set_color((0.2, 0.4, 0.9))

# Set point radius for better visibility
smoke_cloud.set_radius(0.007)

# Register the box mesh
box_mesh = ps.register_surface_mesh("box", default_box_verts, default_box_faces, smooth_shade=False)
box_mesh.set_transparency(0.3)
box_mesh.set_color((0.8, 0.8, 0.8))

# View the smoke particles and box in the 3D UI
ps.show()