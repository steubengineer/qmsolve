import numpy as np
import random as rand
from qmsolve import Hamiltonian, SingleParticle, TimeSimulation, init_visualization, femtoseconds, m_e, Å

#lattice site array
s = 8.0
so = s/2.0
nx = 10
ny = 15

σ1 = 1.5
dσ = 0.5

nmag = 0.5
rand.seed(2112)
def urand():
    return rand.uniform(-nmag, nmag)

#generate a uniform lattice with uniform σ
x_point, y_point = np.meshgrid(np.linspace(0, nx*s*2*np.sqrt(2.0)/2., nx),
                               np.linspace(-ny*s, ny*s, ny))
σ_point = np.full(x_point.shape, σ1)

#shift every other row to make a hex lattice; vary σ per row
for i in range(0, ny):
    for j in range(0, nx):
        if j % 2 == 0:
            y_point[i,j] += so
            if i % 2 == 0:
                σ_point[i,j] += dσ
        else:
            y_point[i, j] -= so

        x_point[i,j] += urand()
        y_point[i,j] += urand()

#lattice point functional
def lattice_point(x,y, x0,y0, σ0):
    V0 = 40
    return V0*np.exp( -1/(4* σ0**2) * ((x-x0)**2+(y-y0)**2))

#interaction potential
def lattice(particle):
    V = 0
    for i in range(0,ny):
        for j in range(0,nx):
            V += lattice_point(particle.x,particle.y,
                               x_point[i,j],y_point[i,j],
                               σ_point[i,j])
    return V

#build the Hamiltonian of the system
H = Hamiltonian(particles = SingleParticle(m = m_e),
                potential = lattice, 
                spatial_ndim = 2, N = 1000, extent = 50 * Å)


# Define the wavefunction at t = 0  (initial condition)
def initial_wavefunction(particle):
    #This wavefunction correspond to a gaussian wavepacket with a mean X momentum equal to p_x0
    σ = 1.0 * Å
    v0 = 20 * Å / femtoseconds
    p_x0 = 1*m_e * v0
    p_y0 = 1*m_e * v0
    return np.exp( -1/(4* σ**2) * ((particle.x+15)**2+(particle.y+15)**2)) / np.sqrt(2*np.pi* σ**2) * np.exp(p_x0*particle.x*1j + p_y0*particle.y*1j)



# Set and run the simulation
total_time = 0.75 * femtoseconds
sim = TimeSimulation(hamiltonian=H, method="split-step-cupy")
sim.run(initial_wavefunction, total_time=total_time, dt=total_time/1200, store_steps=300)


# Finally, we visualize the time dependent simulation
vis = init_visualization(sim)
vis.animate(xlim=[-25* Å,25* Å], ylim=[-25* Å,25* Å],
            potential_saturation=1.0, wavefunction_saturation=0.2,
            animation_duration=5, fps=60, save_animation=True
            )
vis.plot(t=total_time, xlim=[-25*Å, 25*Å], ylim=[-25*Å, 25*Å],
         potential_saturation=1.0, wavefunction_saturation=0.1
         )