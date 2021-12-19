# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=True

import numpy as np
#cimport numpy as np
#cimport random as cyrandom
import cython.random as cyrandom
from vector cimport vector
from libc.math cimport fabs
from types cimport Model, Delay, Propensity, Rule
from scipy.integrate import odeint, ode
import sys
import warnings
import logging
#from pyprob.distributions import Categorica
#from pyprob import Model
#from pyprob.distributions import Categorical, Normal, Uniform


##################################################                ####################################################
######################################              DELAY QUEUE TYPES                   ##############################
#################################################                     ################################################

cdef class DelayQueue:
    cdef void add_reaction(self, double time, unsigned rxn_id, double amount):
        """
        Add a reaction to the queue.
        :param time: (double) The time at which the reaction will occur
        :param rxn_id: (unsigned) The ID of the reaction, i.e. its column index in the stoich matrix
        :param amount: (double) How many of the reaction occurs?
        :return: None
        """

        pass

    def py_add_reaction(self,double time, unsigned rxn_id, double amount):
        self.add_reaction(time, rxn_id, amount)


    cdef double get_next_queue_time(self):
        """
        Find the nearest queued time. Note that it's possible no reaction occurs at the next queue time.
        This is possible in the case where the queue internally updates with some time resolution even if no reactions
        happen.
        :return: (double) The next queue time.
        """

        return 0.0

    def py_get_next_queue_time(self):
        return self.get_next_queue_time()

    cdef void get_next_reactions(self, double *rxn_array):
        """
        Find the next reaction time. rxn_array must have at least enough room available for the number of reactions.

        :param rxn_array: (double *) A place to store how many of each reaction occurs at the next queued time.
        :return: None
        """
        pass

    def py_get_next_reactions(self, np.ndarray[np.double_t, ndim=1] rxn_array):
        self.get_next_reactions(<double*> rxn_array.data)


    cdef void advance_time(self):
        """
        Advance the queue to the next queue relevant time and perform whatever internal updates are necessary. Make
        sure to call get_next_reactions() before advance_time() or you will never know what reactions occurred.
        :return: None
        """
        pass

    def py_advance_time(self):
        self.advance_time()


    cdef DelayQueue copy(self):
        """
        Cope the DelayQueue and return a new totally independent but duplicate one.
        :return: (DelayQueue) The copied DelayQueue
        """
        return None

    def py_copy(self):
        return self.copy()


    cdef void set_current_time(self, double t):
        """
        Set the current time for the queue.
        :param t: (double) the time.
        :return: None
        """
        pass

    def py_set_current_time(self, double t):
        self.set_current_time(t)

    cdef DelayQueue clear_copy(self):
        """
        Copy the DelayQueue and return a new one with the same config, but with no reactions contained in it.
        :return: (DelayQueue) A new and clear DelayQueue.
        """
        return None

    def py_clear_copy(self):
        return self.clear_copy()

    cdef np.ndarray binomial_partition(self, double p):
        """
        Partition the delay queue into two delay queues with reactions switching according to probability p
        :param p: (double) The binomial parameter 0 < p < 1
        :return: (np.ndarray) A length 2 array of objects. Each of these has to be casted back to a DelayQueue type
        """
        return None
    def py_binomial_partition(self, double p):
        return self.binomial_partition(p)

cdef class ArrayDelayQueue(DelayQueue):
    def __init__(self, np.ndarray queue, double dt, double current_time):
        """
        Initialize with a queue, dt resolution, and current time. The queue should have one row for each reaction and
        the max future time that can be handled is dt*(number of queue columns).

        :param queue: (np.ndarray) 2-D array containing the current time.
        :param dt: (double) The time resolution dt
        :param current_time: (double) The current time.
        """
        self.num_reactions = queue.shape[0]
        self.num_cols = queue.shape[1]
        self.queue = queue
        self.next_queue_time = current_time + dt
        self.dt = dt
        self.start_index = 0

    @staticmethod
    def setup_queue(unsigned num_reactions, unsigned queue_length, double dt):
        """
        Static method to create an empty ArrayDelayQueue given a desired length, number of reactions, and dt.

        :param num_reactions: (unsigned) number of reactions in the system
        :param queue_length: (unsigned) length of the queue
        :param dt: (double) time step
        :return: The created array delay queue object
        """
        return ArrayDelayQueue(np.zeros((num_reactions,queue_length)), dt, 0.0)


    cdef DelayQueue copy(self):
        cdef ArrayDelayQueue a = ArrayDelayQueue(self.queue, self.dt, 0.0)
        a.num_reactions = self.num_reactions
        a.num_cols = self.num_cols
        a.next_queue_time = self.next_queue_time
        a.dt = self.dt
        a.start_index = self.start_index
        a.queue = self.queue.copy()
        return a

    cdef DelayQueue clear_copy(self):
        """
        Copy the DelayQueue and return a new one with the same config, but with no reactions contained in it.
        :return: (DelayQueue) A new and clear DelayQueue.
        """
        cdef ArrayDelayQueue a = ArrayDelayQueue(self.queue, self.dt, 0.0)
        a.num_reactions = self.num_reactions
        a.num_cols = self.num_cols
        a.next_queue_time = self.next_queue_time
        a.dt = self.dt
        a.start_index = self.start_index
        a.queue = np.zeros((self.num_reactions,self.num_cols))
        return a


    cdef void set_current_time(self, double t):
        """
        Set the current time
        :param t: (double) the time
        :return: None
        """
        self.next_queue_time = t + self.dt

    cdef void add_reaction(self, double time, unsigned rxn_id, double amount):
        """
        Add a reaction to the queue. If the reaction time is past the max time supported by the queue length, then
        truncate to the maximum queue time. Round to the nearest dt grid point as well when inserting.
        :param time: (double) the time at which the reaction occurs
        :param rxn_id: (unsigned) the id of the reaction
        :param amount: (double) how many of the reaction occurs (typically 1.0)
        :return:
        """
        # Round to the nearest entry in the delay queue.
        cdef int index = int( (time - self.next_queue_time) / self.dt + 0.5 )
        # Don't let the index get too small or too big, truncate to fit into the queue
        if index < 0:
            index = 0
        elif index >= int(self.num_cols):
            index = self.num_cols-1

        # Shift by the start index offset
        index = (index + self.start_index) % self.num_cols

        self.queue[rxn_id,index] += amount

    cdef double get_next_queue_time(self):
        return self.next_queue_time

    cdef void get_next_reactions(self, double *rxn_array):
        cdef unsigned i
        for i in range(self.num_reactions):
            rxn_array[i] = self.queue[i,self.start_index]



    cdef np.ndarray binomial_partition(self, double p):
        """
        Partition the delay queue into two delay queues with reactions switching according to probability p
        :param p: (double) The binomial parameter 0 < p < 1
        :return: (np.ndarray) A length 2 array of objects. Each of these has to be casted back to a DelayQueue type
        """
        cdef ArrayDelayQueue q1 = self.clear_copy()
        cdef ArrayDelayQueue q2 = self.clear_copy()

        cdef unsigned time_points = q1.queue.shape[1]
        cdef unsigned num_reactions = q1.queue.shape[0]

        cdef unsigned time_index = 0
        cdef unsigned reaction_index = 0

        for time_index in range(time_points):
            for reaction_index in range(num_reactions):
                q1.queue[reaction_index,time_index] = cyrandom.binom_rnd_f(self.queue[reaction_index,time_index],p)
                q2.queue[reaction_index,time_index] = self.queue[reaction_index,time_index] - q1.queue[reaction_index,time_index]

        cdef np.ndarray a = np.empty(2,dtype=object)

        a[0] = q1
        a[1] = q2

        return a


    cdef void advance_time(self):
        # advance time by dt
        self.next_queue_time += self.dt
        # clear the current next queued time fully
        cdef unsigned i
        for i in range(self.num_reactions):
            self.queue[i,self.start_index] = 0
        # advanced the start index by 1 cycling around the end.
        self.start_index = (self.start_index + 1) % self.num_cols

##################################################                ####################################################
######################################              SIMULATION INTERFACES               ##############################
#################################################                     ################################################

cdef class CSimInterface:
    cdef np.ndarray get_update_array(self):
        return self.update_array

    def py_get_update_array(self):
        return self.get_update_array()


    cdef np.ndarray get_delay_update_array(self):
        return self.delay_update_array
    def py_get_delay_update_array(self):
        return self.get_delay_update_array()

    #Checks model or interface is valid. Meant to be overriden by the subclass
    cdef void check_interface(self):
        logging.info("No interface Checking Implemented")
    # meant to be overriden by the subclass
    cdef double compute_delay(self, double *state, unsigned rxn_index):
        return 0.0

    # must be overriden by subclass
    cdef void compute_propensities(self, double *state, double *propensity_destination, double time):
        pass
    cdef void compute_volume_propensities(self, double *state, double *propensity_destination, double volume, double time):
        self.compute_propensities(state, propensity_destination, time)

    # by default stochastic propensities are assumed to be the same as normal propensities. This may be overwritten by the subclass, however.
    cdef void compute_stochastic_propensities(self, double *state, double *propensity_destination, double time):
        self.compute_propensities(state, propensity_destination, time)

    # by default stochastic propensities are assumed to be the same as normal propensities. This may be overwritten by the subclass, however.
    cdef void compute_stochastic_volume_propensities(self, double *state, double *propensity_destination, double volume, double time):
        self.compute_volume_propensities(state, propensity_destination, volume, time)

    cdef unsigned requires_delay(self):
        return self.delay_flag

    cdef np.ndarray get_initial_state(self):
        return self.initial_state

    def py_get_initial_state(self):
        return self.get_initial_state()

    cdef void set_initial_state(self, np.ndarray a):
        self.initial_state = a

    def py_set_initial_state(self, np.ndarray a):
        self.set_initial_state(a)

    cdef unsigned get_num_reactions(self):
        return self.num_reactions

    def py_get_num_reactions(self):
        return self.get_num_reactions()

    cdef unsigned get_num_species(self):
        return self.num_species

    def py_get_num_species(self):
        return self.get_num_species()

    cdef double get_initial_time(self):
        return self.initial_time

    def py_get_initial_time(self):
        return self.get_initial_time()

    cdef void set_initial_time(self, double t):
        self.initial_time = t

    def py_set_initial_time(self, double t):
        self.set_initial_time(t)

    cdef void set_dt(self, double dt):
        self.dt = dt

    def py_set_dt(self, double dt):
        self.set_dt(dt)

    cdef double get_dt(self):
        return self.dt

    def py_get_dt(self):
        return self.get_dt()

    cdef double* get_param_values(self):
        return <double*> 0

    def py_get_param_values(self):
        return None

    cdef unsigned get_num_parameters(self):
        return 0

    def py_get_num_parameters(self):
        return self.get_num_parameters()

    cdef unsigned get_number_of_rules(self):
        return 0

    def py_get_number_of_rules(self):
        return self.get_number_of_rules()

    cdef void apply_repeated_rules(self, double *state, double time, unsigned rule_step):
        pass

    cdef void apply_repeated_volume_rules(self, double *state, double volume, double time, unsigned rule_step):
        pass

    def py_apply_repeated_rules(self, np.ndarray[np.double_t, ndim=1] state, double time=0.0, rule_step = True):
        self.apply_repeated_rules(<double*> state.data,time, rule_step)

    def py_apply_repeated_volume_rules(self, np.ndarray[np.double_t, ndim=1] state, double volume = 1.0, double time=0.0, rule_step = True):
        self.apply_repeated_volume_rules(<double*> state.data, volume, time, rule_step)


    # Prepare for determinsitic simulation by creating propensity buffer and also doing the compressed stoich matrix
    cdef void prep_deterministic_simulation(self):
        # Clear out the vectors
        self.S_indices.clear()
        self.S_values.clear()
        cdef unsigned r
        cdef unsigned s
        # keep track of nonzero indices and the coefficients as well
        for s in range(self.num_species):
            # Add vectors for that row
            self.S_indices.push_back(vector[int]())
            self.S_values.push_back(vector[int]())
            for r in range(self.num_reactions):
                if self.update_array[s,r]+self.delay_update_array[s,r] != 0:
                    self.S_indices[s].push_back(r)
                    self.S_values[s].push_back(self.update_array[s,r]+self.delay_update_array[s,r])
        # Create proper size propensity buffer.
        self.propensity_buffer = np.zeros(self.num_reactions,)

        # Set the global simulation object to this model
        global global_sim
        global_sim = self

    def py_prep_deterministic_simulation(self):
        self.prep_deterministic_simulation()

    # Compute deterministic derivative
    cdef void calculate_deterministic_derivative(self, double *x, double *dxdt, double t):
        # Get propensities before doing anything else.
        cdef double *prop = <double*> (self.propensity_buffer.data)
        self.compute_propensities(x,  prop, t)

        cdef unsigned s
        cdef unsigned j
        for s in range(self.num_species):
            dxdt[s] = 0;
            for j in range(self.S_indices[s].size()):
                dxdt[s] += prop[ self.S_indices[s][j]  ] * self.S_values[s][j]


    def py_calculate_deterministic_derivative(self, np.ndarray[np.double_t,ndim=1] x, np.ndarray[np.double_t,ndim=1] dx,
                                              double t):
        self.calculate_deterministic_derivative(<double*> x.data, <double*> dx.data, t)



cdef class ModelCSimInterface(CSimInterface):
    def __init__(self, external_model):
        self.model = external_model
        #Check Model and initialization
        if not self.model.initialized:
            self.model.py_initialize()
            logging.info("Uninitialized Model Passed into ModelCSimInterface. Model.py_initialize() called automatically.")
        self.check_interface()
        self.c_propensities = self.model.get_c_propensities()
        self.c_delays = self.model.get_c_delays()
        self.c_repeat_rules = self.model.get_c_repeat_rules()
        self.update_array = self.model.get_update_array()
        self.delay_update_array = self.model.get_delay_update_array()
        self.initial_state = self.model.get_species_values()
        self.np_param_values = self.model.get_params_values()
        self.c_param_values = <double*>(self.np_param_values.data)
        self.num_reactions = self.update_array.shape[1]
        self.num_species = self.update_array.shape[0]
        self.dt = 0.01

    cdef unsigned get_number_of_species(self):
        return self.num_species

    cdef unsigned get_number_of_reactions(self):
        return self.num_reactions
        
    cdef void check_interface(self):
        if not self.model.initialized:
            raise RuntimeError("Model has been changed since CSimInterface instantiation. CSimInterface no longer valid.")

    cdef double compute_delay(self, double *state, unsigned rxn_index):
        return  (<Delay> (self.c_delays[0][rxn_index])).get_delay(state, self.c_param_values)

    cdef void compute_propensities(self, double *state, double *propensity_destination, double time):
        cdef unsigned rxn
        for rxn in range(self.num_reactions):
            propensity_destination[rxn] = (<Propensity> (self.c_propensities[0][rxn]) ).get_propensity(state, self.c_param_values, time)

    cdef void compute_volume_propensities(self, double *state, double *propensity_destination, double volume, double time):
        cdef unsigned rxn
        for rxn in range(self.num_reactions):
            propensity_destination[rxn] = (<Propensity> (self.c_propensities[0][rxn]) ).get_volume_propensity(state, self.c_param_values,
                                                                                                              volume, time)
    cdef void compute_stochastic_propensities(self, double *state, double *propensity_destination, double time):
        cdef unsigned rxn
        for rxn in range(self.num_reactions):
            propensity_destination[rxn] = (<Propensity> (self.c_propensities[0][rxn]) ).get_stochastic_propensity(state,
                                                                                                       self.c_param_values, time)
    cdef void compute_stochastic_volume_propensities(self, double *state, double *propensity_destination, double volume, double time):
        cdef unsigned rxn
        for rxn in range(self.num_reactions):
            propensity_destination[rxn] = (<Propensity> (self.c_propensities[0][rxn]) ).get_stochastic_volume_propensity(state, self.c_param_values, volume, time)

    cdef unsigned get_number_of_rules(self):
        return self.c_repeat_rules[0].size()

    cdef void apply_repeated_rules(self, double *state, double time, unsigned rule_step):
        cdef unsigned rule_number
        for rule_number in range(self.c_repeat_rules[0].size()):
            (<Rule> (self.c_repeat_rules[0][rule_number])).execute_rule(state, self.c_param_values, time, self.dt, rule_step)

    cdef void apply_repeated_volume_rules(self, double *state, double volume, double time, unsigned rule_step):
        cdef unsigned rule_number
        for rule_number in range(self.c_repeat_rules[0].size()):
            (<Rule> (self.c_repeat_rules[0][rule_number])).execute_volume_rule(state, self.c_param_values, volume, time, self.dt, rule_step)

    cdef np.ndarray get_initial_state(self):
        return self.initial_state

    cdef void set_initial_state(self, np.ndarray a):
        np.copyto(self.initial_state,a)

    cdef double* get_param_values(self):
        return self.c_param_values

    cdef void set_param_values(self, np.ndarray params):
        self.np_param_values = params
        self.c_param_values = <double*>(self.np_param_values.data)

    def py_get_param_values(self):
        return self.np_param_values

    def py_set_param_values(self, params):
        if len(params) != len(self.np_param_values):
            raise ValueError(f"params must be a numpy array of length {len(self.np_param_values)}. Recieved {params}.")
        self.np_param_values = params
        self.c_param_values = <double*>(self.np_param_values.data)


    cdef unsigned get_num_parameters(self):
        return self.np_param_values.shape[0]

cdef class SSAResult:
    def __init__(self, np.ndarray timepoints, np.ndarray result):
        self.timepoints = timepoints
        self.simulation_result = result

    def py_get_timepoints(self):
        return self.get_timepoints()

    def py_get_result(self):
        return self.get_result()

    #Returns a Pandas Data Frame, if it is installed. If not, a Numpy array is returned.
    def py_get_dataframe(self, Model = None):
        try:
            import pandas
            if Model == None:
                warnings.warn("No Model passed into py_get_dataframe. No species names will be attached to the data frame.")
                df = pandas.DataFrame(data = self.get_result())
            else:
                columns = Model.get_species_list()
                df = pandas.DataFrame(data = self.get_result(), columns = columns)
            df['time'] = self.timepoints
            return df

        except ModuleNotFoundError:
            warnings.warn("py_get_dataframe requires the pandas Module to return a Pandas Dataframe object. Numpy array being returned instead.")
            return self.py_get_result()


    def py_empirical_distribution(self, start_time = None, species = None, Model = None, final_time = None, max_counts = None):
        """
            calculates the empirical distribution of a trajectory over counts
            start_time: the time to begin the empirical calculation
            final_time: time to end the empirical marginalization
            species: the list of species inds or names to calculate over. Marginalizes over non-included species
            max_counts: a list (size N-species) of the maximum count expected for each species. 
                 If max_counts[i] == 0, defaults to the maximum count found in the simulation: max(results[:, i]).
                 Useful for getting distributions of a specific size/shape.
            Model: the model used to produce the results. Required if species are referenced by name isntead of index
        """
        if species is None:
            species_inds = [i for i in range(self.simulation_result.shape[1])]
        else:
            species_inds = []
            for s in species:
                if isinstance(s, int):
                    species_inds.append(s)
                elif isinstance(s, str) and Model is None:
                    raise ValueError("Must pass in a Model along with species list if using species' names.")
                elif isinstance(s, str) and s in Model.get_species2index():
                    species_inds.append(Model.get_species2index()[s])
                else:
                    raise ValueError(f"Unknown species {s}.")

        if start_time is None:
            start_time = self.timepoints[0]
        elif start_time < self.timepoints[0]:
            raise ValueError(f"final_time={start_time} is greater than simulation_start={self.timepoints[0]}")

        if final_time is None:
            final_time = self.timepoints[-1]
        elif final_time > self.timepoints[-1]:
            raise ValueError(f"final_time={final_time} is greater than simulation_time={self.timepoints[-1]}")

        if max_counts is None:
            max_counts = [0 for s in species_inds]
        elif len(max_counts) != len(species_inds):
            raise ValueError("max_counts must be a list of the same length as species")

        return self.empirical_distribution(start_time, species_inds, final_time, max_counts)

    #calculates the empirical distribution of a trajectory over counts
    #   start_time: the time to begin the empirical calculation
    #   final_time: time to end the empirical marginalization
    #   species_inds: the list of species inds to calculate over. Marginalizes over non-included inds
    #   max_counts: a list (size N-species) of the maximum count expected for each species. 
    #         If max_counts[i] == 0, defaults to the maximum count found in the simulation: max(results[:, i]).
    #         Useful for getting distributions of a specific size/shape.
    cdef np.ndarray empirical_distribution(self, double start_time, list species_inds, double final_time, list max_counts_list):
        cdef unsigned s, t, ind, prod
        cdef unsigned tstart = len(self.timepoints[self.timepoints < start_time])
        cdef unsigned tend = len(self.timepoints[self.timepoints <= final_time])
        cdef unsigned N_species = len(species_inds)
        cdef double dP = 1./(tend - tstart)
        cdef np.ndarray[np.int_t, ndim=1] index_ar = np.zeros(N_species, dtype = np.int_) #index array
        cdef np.ndarray[np.double_t, ndim = 1] dist
        cdef np.ndarray[np.int_t, ndim = 1] max_counts = np.zeros(N_species, dtype = np.int_) #max species counts

        #Calculate max species counts
        for i in range(N_species):
            s = species_inds[i]
            if max_counts_list[i] == 0: #the maximum number of each species is set for all 0 max species
                max_counts[i] = np.amax(self.simulation_result[tstart:, s], 0)+1
            else:
                max_counts[i] += max_counts_list[i]+1

        #dist = np.zeros(tuple(max_counts.astype(np.int_)+1))#store the distribution here
        dist = np.zeros(np.prod(max_counts)) #Flattened array

        for t in range(tstart, tend, 1):
            #Code for Flat dist arrays
            prod = 1 #a product to represent the size of different dimensions of the flattened array
            ind = 0
            for i in range(N_species, 0, -1):#Go through species index backwards
                s = species_inds[i-1] 
                if self.simulation_result[t, s] > max_counts[i-1]:
                    raise RuntimeError("Encountered a species count greater than max_counts!")
                ind = ind + prod*<np.int_t>self.simulation_result[t, s]
                prod = prod * max_counts[i-1] #update the product for the next index
            dist[ind] = dist[ind] + dP

        return np.reshape(dist, tuple(max_counts))

    #Python wrapper of a fast cython function to compute the first moment (mean) of a set of Species
    def py_first_moment(self, start_time = None, species = None, Model = None, final_time = None):
        """
            calculates the first moment (mean) of a trajectory over counts
            start_time: the time to begin the empirical calculation
            final_time: time to end the empirical marginalization
            species: the list of species inds or names to calculate over. Marginalizes over non-included species
            Model: the model used to produce the results. Required if species are referenced by name isntead of index
        """

        if species is None:
            species_inds = [i for i in range(self.simulation_result.shape[1])]
        else:
            species_inds = []
            for s in species:
                if isinstance(s, int):
                    species_inds.append(s)
                elif isinstance(s, str) and Model is None:
                    raise ValueError("Must pass in a Model along with species list if using species' names.")
                elif isinstance(s, str) and s in Model.get_species2index():
                    species_inds.append(Model.get_species2index()[s])
                else:
                    raise ValueError(f"Unknown species {s}.")

        if start_time is None:
            start_time = self.timepoints[0]
        elif start_time < self.timepoints[0]:
            raise ValueError(f"final_time={start_time} is greater than simulation_start={self.timepoints[0]}")

        if final_time is None:
            final_time = self.timepoints[-1]
        elif final_time > self.timepoints[-1]:
            raise ValueError(f"final_time={final_time} is greater than simulation_time={self.timepoints[-1]}")

        return self.first_moment(start_time, final_time, species_inds)

    #Computes the first moment (average) of all species in the list species_inds
    cdef np.ndarray first_moment(self, double start_time, double final_time, list species_inds):
        cdef unsigned s, i, t
        cdef unsigned tstart = len(self.timepoints[self.timepoints < start_time])
        cdef unsigned tend = len(self.timepoints[self.timepoints <= final_time])
        cdef unsigned N_species = len(species_inds)
        cdef np.ndarray[np.double_t, ndim = 1] means = np.zeros(N_species)

        for i in range(N_species):
            s = species_inds[i]
            for t in range(tstart, tend, 1):
                means[i] += self.simulation_result[t, s]
            means[i] = means[i]/(tend - tstart)

        return means


    #Python wrapper of a fast cython function to compute the standard deviation of a set of Species
    def py_standard_deviation(self, start_time = None, species = None, Model = None, final_time = None):
        """
            calculates the standard deviation of a trajectory over counts
            start_time: the time to begin the empirical calculation
            final_time: time to end the empirical marginalization
            species: the list of species inds or names to calculate over. Marginalizes over non-included species
            Model: the model used to produce the results. Required if species are referenced by name isntead of index
        """
        if species is None:
            species_inds = [i for i in range(self.simulation_result.shape[1])]
        else:
            species_inds = []
            for s in species:
                if isinstance(s, int):
                    species_inds.append(s)
                elif isinstance(s, str) and Model is None:
                    raise ValueError("Must pass in a Model along with species list if using species' names.")
                elif isinstance(s, str) and s in Model.get_species2index():
                    species_inds.append(Model.get_species2index()[s])
                else:
                    raise ValueError(f"Unknown species {s}.")

        if start_time is None:
            start_time = self.timepoints[0]
        elif start_time < self.timepoints[0]:
            raise ValueError(f"final_time={start_time} is greater than simulation_start={self.timepoints[0]}")

        if final_time is None:
            final_time = self.timepoints[-1]
        elif final_time > self.timepoints[-1]:
            raise ValueError(f"final_time={final_time} is greater than simulation_time={self.timepoints[-1]}")

        return self.standard_deviation(start_time, final_time, species_inds)

    #Computes the standard deviation of all species in the list species_inds
    cdef np.ndarray standard_deviation(self, double start_time, double final_time, list species_inds):
        cdef unsigned s, i, t
        cdef unsigned tstart = len(self.timepoints[self.timepoints < start_time])
        cdef unsigned tend = len(self.timepoints[self.timepoints <= final_time])
        cdef unsigned N_species = len(species_inds)
        cdef np.ndarray[np.double_t, ndim = 1] stds = np.zeros(N_species)
        cdef np.ndarray[np.double_t, ndim = 1] means = self.first_moment(start_time, final_time, species_inds)

        for i in range(N_species):
            s = species_inds[i]
            for t in range(tstart, tend, 1):
                stds[i] += (self.simulation_result[t, s]-means[i])**2
            stds[i] = (stds[i]/(tend - tstart))**(1./2.)

        return stds


    #Computes the correlations between species1 and species2
    def py_correlations(self, start_time = None, species1 = None, species2 = None, final_time = None, Model = None):
        """
            calculates the pairwise correlations (species1 x species2) of a trajectory over counts
            start_time: the time to begin the empirical calculation
            final_time: time to end the empirical marginalization
            species1: a list of species names or indices. If None, defaults to all species.
            species2: a second list of species indices or names. All pairs between species1 and species2 are computed
            Model: the model used to produce the results. Required if species are referenced by name isntead of index
        """

        if species1 is None:
            species_inds1 = [i for i in range(self.simulation_result.shape[1])]
            species_inds2 = [i for i in range(self.simulation_result.shape[1])]
        else:
            species_inds1 = []
            for s in species1:
                if isinstance(s, int):
                    species_inds1.append(s)
                elif isinstance(s, str) and Model is None:
                    raise ValueError("Must pass in a Model along with species list if using species' names.")
                elif isinstance(s, str) and s in Model.get_species2index():
                    species_inds1.append(Model.get_species2index()[s])
                else:
                    raise ValueError(f"Unknown species {s}.")

            species_inds2 = []
            for s in species2:
                if isinstance(s, int):
                    species_inds2.append(s)
                elif isinstance(s, str) and Model is None:
                    raise ValueError("Must pass in a Model along with species list if using species' names.")
                elif isinstance(s, str) and s in Model.get_species2index():
                    species_inds2.append(Model.get_species2index()[s])
                else:
                    raise ValueError(f"Unknown species {s}.")

        if start_time is None:
            start_time = self.timepoints[0]
        elif start_time < self.timepoints[0]:
            raise ValueError(f"final_time={start_time} is greater than simulation_start={self.timepoints[0]}")

        if final_time is None:
            final_time = self.timepoints[-1]
        elif final_time > self.timepoints[-1]:
            raise ValueError(f"final_time={final_time} is greater than simulation_time={self.timepoints[-1]}")

        return self.correlations(start_time, final_time, species_inds1, species_inds2)

    #Computes the second moment of between the species in species_inds1 and species_inds2
    cdef np.ndarray correlations(self, double start_time, double final_time, list species_inds1, list species_inds2):
        cdef unsigned s1, s2, i1, i2, t
        cdef unsigned tstart = len(self.timepoints[self.timepoints < start_time])
        cdef unsigned tend = len(self.timepoints[self.timepoints <= final_time])
        cdef unsigned N_species1 = len(species_inds1)
        cdef unsigned N_species2 = len(species_inds2)
        cdef np.ndarray[np.double_t, ndim = 2] cors = np.zeros((N_species1, N_species2))
        cdef np.ndarray[np.double_t, ndim = 1] means1 = self.first_moment(start_time, final_time, species_inds1)
        cdef np.ndarray[np.double_t, ndim = 1] means2 = self.first_moment(start_time, final_time, species_inds2)
        cdef np.ndarray[np.double_t, ndim = 1] standard_devs1 = self.standard_deviation(start_time, final_time, species_inds1)
        cdef np.ndarray[np.double_t, ndim = 1] standard_devs2 = self.standard_deviation(start_time, final_time, species_inds2)

        for i1 in range(N_species1):
            s1 = species_inds1[i1]
            for i2 in range(N_species2):
                s2 = species_inds2[i2]
                for t in range(tstart, tend, 1):
                    cors[i1, i2] += (self.simulation_result[t, s1]-means1[i1])*(self.simulation_result[t, s2]-means2[i2])
                cors[i1, i2] = cors[i1, i2]/((tend - tstart)*standard_devs1[i1]*standard_devs2[i2])

        return cors

    #Python wrapper of a fast cython function to compute the second moment (E[S1*S2]) pairwise between two lists of species
    def py_second_moment(self, start_time = None, species1 = None, species2 = None, final_time = None, Model = None):
        """
            calculates the pairwise second moments of a trajectory over counts
            start_time: the time to begin the empirical calculation
            final_time: time to end the empirical marginalization
            species1: a list of species names or indices. If None, defaults to all species.
            species2: a second list of species indices or names. All pairs between species1 and species2 are computed
            Model: the model used to produce the results. Required if species are referenced by name isntead of index
        """

        if species1 is None:
            species_inds1 = [i for i in range(self.simulation_result.shape[1])]
            species_inds2 = [i for i in range(self.simulation_result.shape[1])]
        else:
            species_inds1 = []
            for s in species1:
                if isinstance(s, int):
                    species_inds1.append(s)
                elif isinstance(s, str) and Model is None:
                    raise ValueError("Must pass in a Model along with species list if using species' names.")
                elif isinstance(s, str) and s in Model.get_species2index():
                    species_inds1.append(Model.get_species2index()[s])
                else:
                    raise ValueError(f"Unknown species {s}.")

            species_inds2 = []
            for s in species2:
                if isinstance(s, int):
                    species_inds2.append(s)
                elif isinstance(s, str) and Model is None:
                    raise ValueError("Must pass in a Model along with species list if using species' names.")
                elif isinstance(s, str) and s in Model.get_species2index():
                    species_inds2.append(Model.get_species2index()[s])
                else:
                    raise ValueError(f"Unknown species {s}.")

        if start_time is None:
            start_time = self.timepoints[0]
        elif start_time < self.timepoints[0]:
            raise ValueError(f"final_time={start_time} is greater than simulation_start={self.timepoints[0]}")

        if final_time is None:
            final_time = self.timepoints[-1]
        elif final_time > self.timepoints[-1]:
            raise ValueError(f"final_time={final_time} is greater than simulation_time={self.timepoints[-1]}")

        return self.second_moment(start_time, final_time, species_inds1, species_inds2)

    #Computes the second moment of between the species in species_inds1 and species_inds2
    cdef np.ndarray second_moment(self, double start_time, double final_time, list species_inds1, list species_inds2):
        cdef unsigned s1, s2, i1, i2, t
        cdef unsigned tstart = len(self.timepoints[self.timepoints < start_time])
        cdef unsigned tend = len(self.timepoints[self.timepoints <= final_time])
        cdef unsigned N_species1 = len(species_inds1)
        cdef unsigned N_species2 = len(species_inds2)
        cdef np.ndarray[np.double_t, ndim = 2] moments = np.zeros((N_species1, N_species2))

        for i1 in range(N_species1):
            s1 = species_inds1[i1]
            for i2 in range(N_species2):
                s2 = species_inds2[i2]
                for t in range(tstart, tend, 1):
                    moments[i1, i2] += self.simulation_result[t, s1]*self.simulation_result[t, s2]
                moments[i1, i2] = moments[i1, i2]/(tend - tstart)

        return moments


# Regular simulations with no volume or delay involved.
cdef class RegularSimulator:
    cdef SSAResult simulate(self, CSimInterface sim, np.ndarray timepoints):
        """
        Perform a simple regular stochastic simulation with no delay or volume involved.

        MUST BE SUBCLASSED.
        :param sim: (CSimInterface) The reaction system. Must have time initialized.
        :param timepoints: (np.ndarray) The time points (must be greater than the initial time).
        :return: (SSAResult) The simulation result.
        """
        raise NotImplementedError("simulate function not implemented for RegularSimulator")

    def py_simulate(self, CSimInterface sim, np.ndarray timepoints):
        #suggested that interfaces do some error checking on themselves to prevent kernel crashes.
        sim.check_interface()
        return self.simulate(sim,timepoints)

cdef class SSASimulator(RegularSimulator):
    """
    A class for implementing a stochastic SSA simulator.
    """
    cdef SSAResult simulate(self, CSimInterface sim, np.ndarray timepoints):
        cdef np.ndarray[np.double_t,ndim=1] c_timepoints = timepoints
        cdef np.ndarray[np.double_t,ndim=1] c_current_state = sim.get_initial_state().copy()
        cdef np.ndarray[np.double_t,ndim=2] c_stoich = sim.get_update_array() + sim.get_delay_update_array()
        cdef np.ndarray[np.double_t,ndim=2] c_delay_stoich = sim.get_delay_update_array()

        cdef unsigned num_species = c_stoich.shape[0]
        cdef unsigned num_reactions = c_stoich.shape[1]
        cdef unsigned num_timepoints = len(timepoints)

        cdef double final_time = timepoints[num_timepoints-1]

        cdef double current_time = sim.get_initial_time()
        cdef double proposed_time = 0.0
        cdef double dt = sim.get_dt()
        cdef double Lambda = 0.0
        cdef unsigned reaction_fired = 0

        cdef np.ndarray[np.double_t,ndim=2] c_results = np.zeros((num_timepoints, num_species),dtype=np.double)
        cdef np.ndarray[np.double_t,ndim=1] c_propensity = np.zeros(num_reactions)

        # Now do the SSA part
        cdef unsigned current_index = 0
        cdef unsigned reaction_choice = 0
        cdef unsigned species_index = 0
        cdef unsigned rule_step = 1


        while current_index < num_timepoints:
            # Compute propensity in place
            sim.apply_repeated_rules(<double*> c_current_state.data,current_time, rule_step)
            sim.compute_stochastic_propensities(<double*> c_current_state.data, <double*> c_propensity.data,current_time)
            # Sample the next reaction time and update
            Lambda = cyrandom.array_sum(<double*> c_propensity.data,num_reactions)

            if Lambda == 0:
                proposed_time = c_timepoints[current_index]
                reaction_fired = 0
                rule_step = 1
            else:
                proposed_time = current_time + cyrandom.exponential_rv(Lambda)
                reaction_fired = 1
                rule_step = 0


            #Go to the next reaction or the next timepoint, whichever is closer
            if proposed_time > c_timepoints[current_index]:
                current_time = c_timepoints[current_index]
                reaction_fired = 0
                rule_step = 1
            else:
                current_time = proposed_time

            # Update previous states
            while current_index < num_timepoints and c_timepoints[current_index] <= current_time:
                for species_index in range(num_species):
                    c_results[current_index,species_index] = c_current_state[species_index]
                current_index += 1

            # Choose a reaction and update the state accordingly.
            if Lambda > 0 and reaction_fired:
                #print(Lambda)

                reaction_choice = cyrandom.sample_discrete(num_reactions, <double*> c_propensity.data , Lambda)
                for species_index in range(num_species):
                    c_current_state[species_index] += c_stoich[species_index,reaction_choice]

        return SSAResult(timepoints,c_results)

#A wrapper function to allow easy simulation of Models
def py_simulate_model(timepoints, Model = None, Interface = None, stochastic = False, 
                    delay = None, safe = False, return_dataframe = True, **keywords):
    

    #Check model and interface
    Interface = ModelCSimInterface(Model)

    #check timestep
    dt = timepoints[1]-timepoints[0]
    if not np.allclose(timepoints[1:] - timepoints[:-1], dt):
        warnings.warn("The timestep in timepoints is not uniform! Timepoints should be a linear set of points...but we'll try to simulate anyways.")
    else:
        Interface.py_set_dt(dt)

    Sim = SSASimulator()
    result = Sim.py_simulate(Interface, timepoints)

    if return_dataframe:
        return result.py_get_dataframe(Model = Model)
    else:
        return result

import bioscrape
import numpy as np
model = bioscrape.sbmlutil.import_sbml("//mnt//d//northeastern//research//causal_inference/Vivarium//vivarium-notebooks//notebooks//LacOperon_stochastic.xml")

timepoints = np.arange(0, 10, 1.0)
#results_det = py_simulate_model(timepoints, Model = model) #Returns a Pandas DataFrame

#Simulate the Model Stochastically
results_stoch = bioscrapePPL.simulator.py_simulate_model(timepoints, Model = model, stochastic = True)
