import numpy as np
import matplotlib.pyplot as plt

# Constants
N = 400  # Grid size (NxN)
r_initialisation = 50  # Initial radius where particles are generated
r_limit = 400  # Limit for maximum radius of the cluster
num_particles = 10000  # Number of particles
max_steps = 10000  # Maximum steps for random walk

# Initialize lattice and time of arrival grid
def initialize_lattice(N):
    S = np.zeros((N, N), dtype=int)
    time_of_arrival = np.full((N, N), -1, dtype=int)  # Stores time of particle attachment (-1 for no particle)
    center_x, center_y = N // 2, N // 2
    S[center_x, center_y] = 1  # Seed particle in the center
    time_of_arrival[center_x, center_y] = 0  # Time 0 for the initial seed particle
    return S, time_of_arrival, center_x, center_y

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

# Main DLA simulation with time of arrival recording
def simulate_dla(N, num_particles, r_initialisation, r_limit, max_steps):
    S, time_of_arrival, center_x, center_y = initialize_lattice(N)
    
    for i in range(1, num_particles + 1):
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
                time_of_arrival[particle_x, particle_y] = i  # Record time of arrival
                break  # Particle attached, stop random walk

            # Check if particle is out of bounds based on r_limit
            if np.sqrt((particle_x - center_x) ** 2 + (particle_y - center_y) ** 2) > r_limit:
                break  # Particle out of bounds
            
            steps += 1
        
    return S, time_of_arrival

# Calculate cluster properties
def calculate_cluster_properties(S, time_of_arrival, center_x, center_y):
    # Cluster size
    cluster_size = np.sum(S)
    
    # Mean radius
    x, y = np.where(S == 1)
    distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    mean_radius = np.mean(distances)
    
    # Fractal dimension estimation using scaling relation
    if cluster_size > 0:
        fractal_dimension = np.log(cluster_size) / np.log(mean_radius) if mean_radius > 0 else 0
    else:
        fractal_dimension = 0
    
    return cluster_size, mean_radius, fractal_dimension

# Visualization of the DLA cluster with color coding for time of arrival
def visualize_cluster(S, time_of_arrival):
    plt.figure(figsize=(6, 6))
    plt.imshow(time_of_arrival, cmap='plasma', interpolation='nearest')
    plt.colorbar(label='Time of Arrival')
    plt.title('Diffusion-Limited Aggregation Cluster (Time of Arrival Colors)')
    plt.show()

# Run the DLA simulation
S, time_of_arrival = simulate_dla(N, num_particles, r_initialisation, r_limit, max_steps)

# Calculate cluster properties
cluster_size, mean_radius, fractal_dimension = calculate_cluster_properties(S, time_of_arrival, N//2, N//2)

# Print cluster properties
print(f"Cluster Size: {cluster_size}")
print(f"Mean Radius: {mean_radius:.2f}")
print(f"Fractal Dimension: {fractal_dimension:.2f}")

# Visualize the resulting cluster with time of arrival colors
visualize_cluster(S, time_of_arrival)
