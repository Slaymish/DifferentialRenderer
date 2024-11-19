import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Particle-based Scene Representation
class Particle(nn.Module):
    def __init__(self, num_particles):
        super().__init__()
        # Initialize particle parameters
        self.positions = nn.Parameter(torch.randn(num_particles, 3) * 0.1)
        self.scales = nn.Parameter(torch.ones(num_particles, 3) * 0.05)
        self.colors = nn.Parameter(torch.rand(num_particles, 3))
        self.densities = nn.Parameter(torch.ones(num_particles, 1))

def compute_particle_influence(points, particle_positions, particle_scales):
    # points: [N_points, 3]
    # particle_positions: [N_particles, 3]
    # particle_scales: [N_particles, 3]
    # Output: [N_points, N_particles]
    diff = points.unsqueeze(1) - particle_positions.unsqueeze(0)  # [N_points, N_particles, 3]
    scaled_diff = diff / particle_scales.unsqueeze(0)  # Scale differences
    sq_distance = (scaled_diff ** 2).sum(-1)  # [N_points, N_particles]
    influence = torch.exp(-0.5 * sq_distance)  # Gaussian influence
    return influence  # [N_points, N_particles]

class Camera:
    def __init__(self, width, height, fov=60, device='cpu'):
        self.width = width
        self.height = height
        self.fov = fov
        self.near = 0.1
        self.far = 100.0
        self.device = device
        self.position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        self.target = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
        self.up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
        self.update()

    def to(self, device):
        """Move all camera attributes to the specified device."""
        self.device = device
        self.position = self.position.to(device)
        self.target = self.target.to(device)
        self.up = self.up.to(device)
        self.view_matrix = self.view_matrix.to(device)
        self.projection_matrix = self.projection_matrix.to(device)

    def update(self):
        self.forward = F.normalize(self.target - self.position, dim=0)
        self.right = F.normalize(torch.cross(self.forward, self.up, dim=0), dim=0)
        self.up = torch.cross(self.right, self.forward, dim=0)
        
        # View matrix
        self.view_matrix = torch.eye(4, device=self.device)
        self.view_matrix[:3, :3] = torch.stack([self.right, self.up, -self.forward], dim=1)
        self.view_matrix[:3, 3] = -self.view_matrix[:3, :3] @ self.position

        # Projection matrix
        aspect_ratio = self.width / self.height
        fov_rad = torch.deg2rad(torch.tensor(self.fov, dtype=torch.float32, device=self.device))
        f = 1.0 / torch.tan(fov_rad / 2.0)
        self.projection_matrix = torch.zeros((4, 4), device=self.device)
        self.projection_matrix[0, 0] = f / aspect_ratio
        self.projection_matrix[1, 1] = f
        self.projection_matrix[2, 2] = (self.far + self.near) / (self.near - self.far)
        self.projection_matrix[2, 3] = (2 * self.far * self.near) / (self.near - self.far)
        self.projection_matrix[3, 2] = -1.0

    def get_rays(self):
        i, j = torch.meshgrid(
            torch.linspace(0, self.width - 1, self.width, device=self.device),
            torch.linspace(0, self.height - 1, self.height, device=self.device),
            indexing='ij'
        )
        i, j = i.t(), j.t()

        fov_rad = torch.deg2rad(torch.tensor(self.fov, dtype=torch.float32, device=self.device))
        dirs = torch.stack([
            (i - self.width * 0.5) / self.width * 2 * torch.tan(fov_rad / 2) * (self.width / self.height),
            -(j - self.height * 0.5) / self.height * 2 * torch.tan(fov_rad / 2),
            -torch.ones_like(i)
        ], dim=-1)  # [H, W, 3]
        rays_d = dirs @ self.view_matrix[:3, :3].T  # [H, W, 3]
        rays_o = self.position.expand(rays_d.shape)
        return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

def volume_rendering(colors, densities, deltas):
    alpha = 1.0 - torch.exp(-densities * deltas)
    transmittance = torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1.0 - alpha + 1e-10], dim=1),
        dim=1
    )[:, :-1]
    weights = alpha * transmittance  # [N_rays, num_samples]
    color = torch.sum(weights.unsqueeze(-1) * colors, dim=1)  # [N_rays, 3]
    return color

def render_image(scene_particles, camera, num_samples):
    rays_o, rays_d = camera.get_rays()
    rays_o, rays_d = rays_o.to(scene_particles.positions.device), rays_d.to(scene_particles.positions.device)

    # Sample points along rays
    t_vals = torch.linspace(0.0, 1.0, steps=num_samples, device=rays_o.device)
    z_vals = t_vals.expand(rays_o.shape[0], num_samples)
    points = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1)  # [N_rays, num_samples, 3]
    points_flat = points.reshape(-1, 3)  # [N_rays * num_samples, 3]

    # Compute particle influence
    influence = compute_particle_influence(
        points_flat, scene_particles.positions, scene_particles.scales
    )  # [N_points, N_particles]

    # Compute densities and colors at each point
    densities = influence @ scene_particles.densities  # [N_points, 1]
    colors = influence @ scene_particles.colors  # [N_points, 3]

    # Reshape outputs
    densities = densities.reshape(-1, num_samples)  # [N_rays, num_samples]
    colors = colors.reshape(-1, num_samples, 3)  # [N_rays, num_samples, 3]

    # Volume rendering
    deltas = z_vals[:, 1:] - z_vals[:, :-1]
    delta_inf = 1e10 * torch.ones_like(deltas[:, :1], device=deltas.device)
    deltas = torch.cat([deltas, delta_inf], dim=1)
    color = volume_rendering(colors, densities, deltas)
    image = color.reshape(camera.height, camera.width, 3)
    return image

def train(device):
    # Hyperparameters
    width, height = 64, 64
    num_samples = 64
    num_epochs = 10
    lr = 1e-3
    num_particles = 1000  # Adjust based on scene complexity

    # Initialize particles and optimizer
    scene_particles = Particle(num_particles).to(device)
    optimizer = torch.optim.Adam(scene_particles.parameters(), lr=lr)

    # Load dataset with synthetic data
    dataset = load_dataset(scene_particles, width, height, num_samples)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for data in dataset:
            optimizer.zero_grad()
            camera = data['camera']
            target_image = data['image'].to(device)

            # Render image
            rendered_image = render_image(scene_particles, camera, num_samples)

            # Compute loss
            loss = F.mse_loss(rendered_image.reshape(-1, 3), target_image.reshape(-1, 3))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch}, Loss: {total_loss / len(dataset)}')

    # Render from a new viewpoint
    render_from_new_view(scene_particles, width, height, num_samples)

def render_from_new_view(scene_particles, width, height, num_samples):
    camera = Camera(width, height, device='cuda')  # Initialize camera on 'cuda' device
    camera.position = torch.tensor([0.0, 0.0, -4.0], dtype=torch.float32, device='cuda')
    camera.target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda')
    camera.update()
    image = render_image(scene_particles, camera, num_samples)
    plt.imshow(image.cpu().detach().numpy())
    plt.show()

def load_dataset(scene_particles, width, height, num_samples):
    dataset = []
    angles = range(0, 360, 60)
    for angle in angles:
        camera = Camera(width, height, device='cuda')  # Ensure the camera is on the correct device
        radius = 4.0
        angle_rad = torch.deg2rad(torch.tensor(angle, dtype=torch.float32, device=camera.device))
        camera.position = torch.tensor([
            radius * torch.sin(angle_rad), 
            0.0, 
            radius * torch.cos(angle_rad)
        ], dtype=torch.float32, device=camera.device)
        camera.target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=camera.device)
        camera.update()

        # Render the scene from this camera viewpoint
        with torch.no_grad():
            image = render_image(scene_particles, camera, num_samples)
        dataset.append({'camera': camera, 'image': image})
    return dataset

# Create a known simple scene: a sphere
def create_sphere_particles(num_particles):
    # Generate particles on the surface of a sphere
    phi = torch.acos(1 - 2 * torch.rand(num_particles))
    theta = 2 * torch.pi * torch.rand(num_particles)
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    positions = torch.stack([x, y, z], dim=1)
    positions *= 0.5  # Scale the sphere
    scales = torch.ones_like(positions) * 0.05
    colors = torch.ones(num_particles, 3) * torch.tensor([1.0, 0.0, 0.0])  # Red sphere
    densities = torch.ones(num_particles, 1) * 10.0  # Higher density for visibility
    return positions, scales, colors, densities

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create a simple sphere scene
    num_particles = 1000
    positions, scales, colors, densities = create_sphere_particles(num_particles)

    # Initialize particles with known scene
    scene_particles = Particle(num_particles).to(device)
    scene_particles.positions.data = positions.to(device)
    scene_particles.scales.data = scales.to(device)
    scene_particles.colors.data = colors.to(device)
    scene_particles.densities.data = densities.to(device)

    # Train the model
    train(device)
