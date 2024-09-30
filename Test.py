import numpy as np
import matplotlib.pyplot as plt

# Constants
N = 400  # Grid size (NxN)
r_initialisation = 500  # Initial radius where particles are generated
r_limit = 400  # Limit for maximum radius of the cluster
num_particles = 100000  # Number of particles
max_steps = 100000  # Maximum steps for random walk

# Initialize lattice (NxN grid with all zeros)
def initialize_lattice(N):
    S = np.zeros((N, N), dtype=int)
    center_x, center_y = N // 2, N // 2
    S[center_x, center_y] = 1  # Seed particle in the center
    return S, center_x, center_y

# Generate random position for a new particle based on polar coordinates
def generate_particle(r_initialisation, center_x, center_y):
    angle = np.random.uniform(0, 2 * np.pi)
    particle_x = int(center_x + r_initialisation * np.cos(angle))
    particle_y = int(center_y + r_initialisation * np.sin(angle))
    return particle_x, particle_y

# Random walk motion (up, down, left, right)
def random_walk(particle_x, particle_y):
    movements = np.array([[0, 1], [0, -1], [-1, 0], [1, 0]])  # Directions: Up, Down, Left, Right
    random_direction = np.random.randint(0, 4)  # Random direction
    movement = movements[random_direction]
    return particle_x + movement[0], particle_y + movement[1]

# Check if the particle touches the cluster
def is_near_cluster(S, x, y):
    return np.any(S[max(0, x-1):min(N, x+2), max(0, y-1):min(N, y+2)] == 1)

# Main DLA simulation
def simulate_dla(N, num_particles, r_initialisation, r_limit, max_steps):
    S, center_x, center_y = initialize_lattice(N)
    
    for i in range(num_particles):
        particle_x, particle_y = generate_particle(r_initialisation, center_x, center_y)
        
        steps = 0
        while steps < max_steps:
            particle_x, particle_y = random_walk(particle_x, particle_y)
            
            # Ensure the particle stays within bounds
            particle_x = np.clip(particle_x, 0, N-1)
            particle_y = np.clip(particle_y, 0, N-1)
            
            # Check if the particle is near the cluster
            if is_near_cluster(S, particle_x, particle_y):
                S[particle_x, particle_y] = 1  # Add particle to cluster
                break  # Particle attached, stop random walk

            # Check if particle is out of bounds based on r_limit
            if np.sqrt((particle_x - center_x) ** 2 + (particle_y - center_y) ** 2) > r_limit:
                break  # Particle out of bounds
            
            steps += 1
        
    return S

# Visualization of the DLA cluster
def visualize_cluster(S):
    plt.figure(figsize=(6, 6))
    plt.imshow(S, cmap='binary')
    plt.title('Diffusion-Limited Aggregation Cluster')
    plt.show()

# Run the DLA simulation
S = simulate_dla(N, num_particles, r_initialisation, r_limit, max_steps)

# Visualize the resulting cluster
visualize_cluster(S)
