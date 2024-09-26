import numpy

# Create a NumPy array initialized to 0
S = np.zeros((N, N), dtype=int)
# Example: Setting a particle as part of the cluster
center_x, center_y = N // 2, N // 2 # Seed particle position
#"//" symbol is for integer division
S[center_x, center_y] = 1 # Mark the seed particle as part of the cluster