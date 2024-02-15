import numpy as np
from qmsolve import Hamiltonian, SingleParticle,TimeSimulation, init_visualization, m_e, e, Å, T, eV, femtoseconds

#interaction potential
def constant_magnetic_field(particle):

    Bz = 100000 * T

    B_dot_L =  Bz*(-particle.px @ particle.y * 1.0 + particle.py @ particle.x * 1.0)
    𝜇 = 0.5 # e/(2*m_e)
    paramagnetic_term = -𝜇 * B_dot_L

    d = 0.125 # e**2/(8*m_e)
    diamagnetic_term = d* Bz**2 *(particle.x**2 + particle.y**2)

    magnetic_interaction = diamagnetic_term  + paramagnetic_term

    v0 = 80 * Å / femtoseconds #mean initial velocity of the wavepacket
    R = (m_e*v0 / (e*Bz))/Å #cyclotron radius
    print( (u"classical cyclotron radius = {} angstroms".format("%.2f"  % (R)) ) )

    P = (2*np.pi*m_e/(e*Bz))/ femtoseconds #cyclotron period
    print( (u"classical cyclotron period = {} femtoseconds".format("%.2f"  % (P)) ) )


    return magnetic_interaction


#hamiltonian
H = Hamiltonian(particles = SingleParticle(),
                potential = constant_magnetic_field, potential_type = "matrix",
                spatial_ndim = 2, N = 421, extent = 35 * Å)

#wavefunction
def initial_wavefunction(particle):
    #This wavefunction correspond to a gaussian wavepacket with a mean X momentum equal to p_x0
    σ = 1.0 * Å
    v0 = 80 * Å / femtoseconds
    p_x0 = m_e * v0
    return np.exp( -1/(4* σ**2) * ((particle.x-0)**2+(particle.y-5.0*Å)**2)) / np.sqrt(2*np.pi* σ**2)  *np.exp(p_x0*particle.x*1j)


#run
total_time = 100.0 * femtoseconds
total_steps = 12000
n_to_store = 3600
delta_t = total_time / total_steps

print("running ", total_steps," steps, dt = ", format("%.4f" % (delta_t/femtoseconds)), " fs." )
sim = TimeSimulation(hamiltonian = H, method = "crank-nicolson-cupy")
sim.run(initial_wavefunction, total_time = total_time, dt = delta_t, store_steps = n_to_store)


#visualize
vis = init_visualization(sim)
vis.plot(t = total_time, xlim=[-15* Å,15* Å], ylim=[-15* Å,15* Å], potential_saturation = 0.2, wavefunction_saturation = 0.2)
vis.animate(xlim=[-15* Å,15* Å], ylim=[-15* Å,15* Å], potential_saturation = 0.2, wavefunction_saturation = 0.2, animation_duration = 120, fps = 30, save_animation = True, figsize=(12,12), )
