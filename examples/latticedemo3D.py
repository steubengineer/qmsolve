import os
import numpy as np
import random as rand
from qmsolve import Hamiltonian, SingleParticle, TimeSimulation, init_visualization, femtoseconds, m_e, Å

from mayavi import mlab
from tvtk.util import ctf

#lattice site array
s = 8.0
so = s/2.0
nx = 3
ny = 4+4
nz = 5+4

σ1 = 1.5
dσ = 0.5

#generate a uniform lattice with uniform σ
x_point, y_point, z_point = np.meshgrid(
                                np.linspace(0.0, nx*1.5*s, nx),
                                np.linspace(-ny*s, ny*s, ny),
                                np.linspace(-nz*s, nz*s, nz),
                                indexing='ij')
σ_point = np.full(x_point.shape, σ1)

#shift the middle file by half the spacing
for i in range(0, nx):
    for j in range(0, ny):
        for k in range(0, nz):
            if(i%2==1):
                y_point[i,j,k] += 2*so
                z_point[i,j,k] += 2*so

            elif (j % 2 == 0):
                z_point[i, j, k] += 2*so


#potential function for a single lattice point
def lattice_point(x,y,z, x0,y0,z0, σ0):
    V0 = 40
    return V0*np.exp( -1/(4* σ0**2) * ((x-x0)**2+(y-y0)**2+(z-z0)**2) )


#interaction functional of all particles in the lattice
def lattice(particle):
    V = 0
    for i in range(0,nx):
        for j in range(0,ny):
            for k in range(0, nz):
                V += lattice_point(particle.x,particle.y, particle.z,
                                   x_point[i,j,k],y_point[i,j,k], z_point[i,j,k],
                                   σ_point[i,j,k])
    return V


#build the Hamiltonian of the system
H = Hamiltonian(particles=SingleParticle(m=m_e), potential=lattice, spatial_ndim=3, N=162, extent = 40*Å)


#define the wavefunction at t = 0  (initial condition)
def initial_wavefunction(particle):
    #This wavefunction correspond to a gaussian wavepacket with a mean X momentum equal to p_x0
    σ = 1.0 * Å
    v0 = 20 * Å / femtoseconds
    p_x0 = 1*m_e * v0
    p_y0 = 1*m_e * v0
    return (np.exp( -1/(4* σ**2) * ((particle.x+20)**2+(particle.y+20)**2+(particle.z+0.0)**2)) /
                                    np.sqrt(2*np.pi* σ**2) * np.exp(p_x0*particle.x*1j + p_y0*particle.y*1j))


#set and run the simulation
total_time = 0.8 * femtoseconds
num_steps = 160
sim = TimeSimulation(hamiltonian=H, method="split-step-cupy")
sim.run(initial_wavefunction, total_time=total_time, dt=total_time/1800, store_steps=num_steps)


#interactive visualization at a given time
vis = init_visualization(sim)
vis.plot(t=0.6*femtoseconds, figsize=(1000, 1000),
         potential_saturation=0.1, potential_cutoff=0.015,
         wavefunction_saturation=0.1, wavefunction_cutoff=0.025,
         view_azimuth_angle=150, view_elevation_angle=60,
         save_image=False, filename='testimage.png'
         )

#save some of the snapshots to disk
#note that is is a really inefficient way to do animation - a heavy VTK object gets rebuilt
for idx in range(0, num_steps+1, 20):
    fname = u"img_{}.png".format( "%.3i"  % (idx) )
    vis.plot(index=idx, figsize=(1000, 1000),
             potential_saturation=0.1, potential_cutoff=0.015,
             wavefunction_saturation=0.1, wavefunction_cutoff=0.025,
             view_azimuth_angle=150, view_elevation_angle=60,
             save_image=True, filename=fname
             )


#TODO: make a module for exporting binaries or hd5s
#export for testing
#print(sim.Ψ.shape)
#sim.H.Vgrid.astype('float32').tofile('testV.dat')
#sim.Ψ.astype(np.complex64).tofile('testPhi.dat')
