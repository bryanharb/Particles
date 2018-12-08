# -*- coding: utf-8 -*-
"""
Class file for a Particle object.
@author: bryan
"""

import numpy as np

class Particle():
    """Represents one single particle created at creation_time with a constant
    speed. The position must be  consistent with the other parameters, i.e.
                        x = v * (t-creation_time)
    The active Flag is initialized as True, and becomes False as a consequence
    of a collision with another particle.
    """
    def __init__(self, creation_time, speed, position, t, active = True):
        self.creation_time = creation_time
        self.v = speed
        self.t = t
        self.x = speed * (t - creation_time)
        self.active = active
        
    def __repr__(self):
        return("Particle at position {:2.2f} with speed {:2.2f}".format(self.x, self.v))
        
    def __ge__(self, other_particle):
        """A particle is "bigger" than another one if it's creation time is.
        This inherits the order structure of the creation indices, i.e. among
        2 the one born later is bigger.
        """
        return self.creation_time < other_particle.creation_time
        
    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, x_new):
        if x_new != self.v * (self.t - self.creation_time):
            raise ValueError("""The input parameters are not consistent with 
                             the equation of motion. Consult the docstring
                             for Particle.""")
        else:
            self._x = x_new
            
    @property
    def t(self):
        return self._t
    
    @t.setter
    def t(self, t_new):
        if t_new < 0:
            raise ValueError("Time cannot be negative.")
        else:
            # If time changes, the position is automatically updated
            self._t = t_new
            self._x = self.v * (self.t - self.creation_time)
       
    @property
    def active(self):
        return self._active
    
    @active.setter
    def active(self, new_status):
        self._active = new_status
            
    def destroy(self):
        """If the particle is active, this method sets it to inactive. 
        If the particle is inactive, raises ValueError"""
        if self.active:
            self.active = False
        else:
            raise ValueError("This particle is already inactive.")
        
    def collision(self, other_particle):
        """The particle collides with another particle. Destroys them both,"""
        self.destroy()
        other_particle.destroy()


class System():
    def __init__(self, n_particles = np.inf, stable = False):
        """ Simulates a system of particles uniformly generated at t = 1,2,..,n_particles
        with uniform speed on [0,1]. If two particles collide, they both
        disappear. The stable flag gets turned on once the expected time for the
        next event in the system is infinite."""
        self.active_particles = np.array([])
        self.t = 0
        self.generate_particle()
        self.N = len(self.active_particles)
        self.n_particles = n_particles
        self.stable = stable
        
    def __repr__(self):
        if self.N > 0:
            return "t = " + str(self.t) + " - ACTIVE PARTICLES:" + str(self.N) + " \n MAX DISTANCE: " + str(self.active_particles[0].x)
        else:
            return "EMPTY SYSTEM AT TIME t = " + str(self.t)
        
    def showparticles(self):
        print(self.active_particles)
            
    @property
    def active_particles(self):
        return self.active_particles_
    
    @active_particles.setter
    # When active_particles is set, the setter checks which particles are active
    # and updates the N parameter in the system. It also updates the time for
    # all the particles, hence their position.
    def active_particles(self, new_active_particles):
        active_particles = [particle for particle in new_active_particles if particle.active]
        for particle in new_active_particles:
            particle.t = self.t
        self.active_particles_ = np.array(active_particles)
        self.N = len(self.active_particles)
        
    def generate_particle(self):
        """Generates a new particle at the origin with speed uniformly distributed
        on [0,1]"""
        new_speed = np.random.rand()
        new_particle = Particle(self.t, new_speed, 0, self.t)
        self.active_particles = np.append(self.active_particles, new_particle)
        
    def particle_collision(self, particle1, particle2):
        """Makes two particles collide."""
        particle1.collision(particle2)
        self.active_particles = self.active_particles
        
    def next_creation_time(self):
        """Returns the next creation time for a system at time t:
            - If there are particles left to generate it returns the ceil of t
            - Otherwise, it returns np.inf"""
        if self.t < self.n_particles - 1:
            return (np.ceil(self.t + 1e-4))
        else:
            return(np.inf)
    
    def next_collision(self):
        """If there are more than two particles, returns the time of the next
        collision happening among the particles. The function checks every
        subsequent pair of particles. If the speed of the leading one is higher,
        the collision time of that pair is set to np.inf, otherwise the time
        is set to 
                                 (x1 - x2)/(v2-v1)
        It then returns the minimum among these times.
        
        If the i flag is active, returns the position of the particles in the
        next collision.
        """
        if self.N  < 2:
           return(np.inf, None, None, None)
        else:
            collision_times = np.zeros(self.N-1)
            for i in range(self.N-1):
                particle_1 = self.active_particles[i]
                particle_2 = self.active_particles[i + 1]
                x1, x2 = particle_1.x, particle_2.x
                v1, v2 = particle_1.v, particle_2.v
                if v2 > v1:
                    collision_times[i] = self.t + (x1 - x2)/(v2 - v1)
                else:
                    collision_times[i] = np.inf
            i = np.argmin(collision_times)
            first_collision_time = collision_times[i]
            particle_hitting = self.active_particles[i +1]
            hit_particle = self.active_particles[i]
            return (first_collision_time, particle_hitting, hit_particle, i)
            
        
    def next_event(self): 
        """Generates the next event in the system. If it's a creation, returns
        None, otherwise it returns the index of the particle colliding."""
        creation_time = self.next_creation_time()
        collision_time, particle_hitting, hit_particle, i = self.next_collision()
        if collision_time == np.inf and creation_time == np.inf:
            self.stable = True
        elif collision_time > creation_time:
            self.t = int(creation_time)
            self.generate_particle()   
            return(None)
        else:
            self.t = collision_time
            self.particle_collision(particle_hitting, hit_particle)
            return(i)
   
class Simulator():
    """Class to run simulations on the system"""
    
    def __init__(self):
        pass
    
    def are_particles_escaping(self, n, verbose = False, speed = False):
        """Runs a simulation with n particles. If verbose is active, prints
        the status to screen at every update. The speed flag controls the 
        output of the speeds of single particles. The simulation continues
        untile the system reaches a stable state, i.e. the time before the next
        event is infinity. This can happen if there are zero particles or 
        if the remaining particles have ordered speeds.
        
        Returns True if the system reaches stability with more than 0 particles,
        i.e. at least one particle will reach infinity, False otherwise.
        """
        sys = System(n)
        while not sys.stable:
             i = sys.next_event()
             if verbose:
                print("SYSTEM STATUS")
                if i is not None:
                    print("PARTICLE {:d} COLLIDED WITH PARTICLE {:d}.".format(i+1, i+2))
                print(sys)
                if speed:
                    print(sys.active_particles)
                print("=======================================================")
        if sys.N == 0:
            return(False)
        return(True)
        
        
    def montecarlo_escape(self, n, N_sim):
        """Runs are_particles_escaping on n particles N_sim time and returns
        the probability of at least two escapes to infinity"""
        samples = np.array([self.are_particles_escaping(n) for i in range(N_sim)])
        mean = np.mean(samples)
        return(mean)
        
sim = Simulator()
sim.are_particles_escaping(8, verbose = True, speed = True)

                
            
            
        
        
        
    
        