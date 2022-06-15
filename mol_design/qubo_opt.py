import numpy as np
from sklearn import linear_model
from .utils import *
import dimod
import neal 
import pymc3 as pm
from tqdm.auto import tqdm
from dwave.system.composites import EmbeddingComposite, FixedEmbeddingComposite
from dwave.system.samplers import DWaveSampler
import dimod
from minorminer import find_embedding

class QuboOPT:
    def __init__(self, x, target_func, uncertainty=False, method="sa"):
        self.x = x
        self.target_func = target_func
        self.natoms = x.size(0)
        self.atom_types = x.argmax(dim=1).numpy()
        self.atom_symbols = [atom_type_inv[i] for i in self.atom_types]
        self.vars = {(i,j):0 for i in range(self.natoms) for j in range(i+1, self.natoms)}
        self.coeffs = None
        self.uncertainty = uncertainty
        self.explore_done = False
        self.method = method

        if self.method == "qc":
            self.sampler = DWaveSampler(solver = "Advantage_system4.1")
            bqm, _, _ = self.generate_base_qubo()
            self.embedding = self.get_embedding(bqm)
            self.system = FixedEmbeddingComposite(self.sampler, embedding=self.embedding)
        elif self.method == "sa":
            self.sampler = neal.SimulatedAnnealingSampler()

        self.var_history = []
        self.target_history = []
        self.graph_history = []
        self.mol_history = []
    
    def get_embedding(self, bqm):
        __, target_edgelist, target_adjacency = self.sampler.structure
        source_edgelist = list(bqm.quadratic) + [(v, v) for v in bqm.linear]
        embedding = find_embedding(source_edgelist, target_edgelist)
        return embedding

    def sample(self, random=True, valid=True, fully_connected=True, **kwargs):
        if random and not valid:
            A = {key: np.random.binomial(1,0.5) for key in self.vars}
            A = self.sample_to_matrix(A)
        elif random and valid:
            A = self.solve_base_qubo(fully_connected=fully_connected)
        else:
            multiplier = np.max( np.abs(list(self.coeffs.values())) )
            A = self.solve_qubo(self.coeffs, 
                                fully_connected=fully_connected, 
                                multiplier = multiplier,
                                )
        try:
            graph = xA_to_graph(self.x, A)
        except:
            graph = None
        return A, graph

    def solve_base_qubo(self, fully_connected=False):
        bqm, _, _ = self.generate_base_qubo()
        if self.method == "sa":
            response = self.sampler.sample(bqm)
        elif self.method == "qc":
            response = self.system.sample(bqm, num_reads=100)
        else:
            raise ValueError("Only SA and QC supported as methods")
        for sample in response.samples():
            A = self.sample_to_matrix(sample)
            if edge_matrix_validity(self.x, A, fully_connected):
                return A
        return self.solve_base_qubo()

    def solve_qubo(self, coeffs:dict, fully_connected=False, **kwargs):
        bqm, _, _ = self.generate_qubo(**kwargs)
        if self.method == "sa":
            response = self.sampler.sample(bqm)
        elif self.method == "qc":
            response = self.system.sample(bqm, num_reads=100)
        else:
            raise ValueError("Only SA and QC supported as methods")
        for sample in response.samples():
            A = self.sample_to_matrix(sample)
            if edge_matrix_validity(self.x, A, fully_connected):
                return A
        return self.solve_base_qubo()
    
    def sample_to_matrix(self, sample):
        A = np.zeros((self.natoms, self.natoms))
        for (i,j) in sample:
            A[i,j] = sample[i,j]
            A[j,i] = A[i,j]
        return A

    def generate_base_qubo(self):
        lin = {}
        qua = {}
        for i in range(self.natoms):
            # term 1
            for j in range(i+1, self.natoms):
                join( lin, (i,j), 1-valency[self.atom_symbols[i]] )
            
            # term 2
            for j in range(0, i):
                join( lin, (j,i), 1-valency[self.atom_symbols[i]] )
        
        for i in range(self.natoms):
            # term 3
            for j in range(i+1, self.natoms):
                for l in range(j+1, self.natoms):
                    join(qua, ((i,j), (i,l)), 2)
            
            # term 4
            for j in range(0, i):
                for l in range(0, j):
                    join(qua, ((j,i), (l,i)), 2)

            # term 5
            for j in range(i+1, self.natoms):
                for l in range(0, i):
                    join(qua, ((i,j), (l,i)), 1)
        
        bqm = dimod.BinaryQuadraticModel(lin, qua, 0.0, dimod.BINARY) 
        return bqm, lin, qua

    def generate_qubo(self, multiplier=1):
        _, lin, qua = self.generate_base_qubo()
        for key in lin:
            lin[key] *= multiplier
        for key in qua:
            qua[key] *= multiplier

        coeffs = self.coeffs
        for key in coeffs:
            lin[key] += coeffs[key]

        bqm = dimod.BinaryQuadraticModel(lin, qua, 0.0, dimod.BINARY) 
        return bqm, lin, qua

    def update_coeffs(self):
        x = np.array([list(i.values()) for i in self.var_history])
        y = np.array(self.target_history)
        N = x.shape[-1]

        if self.uncertainty:
            model = pm.Model()
            with model:
                alpha = pm.Normal("alpha", mu=0, sigma=1, shape=N)
                sigma = pm.HalfNormal("sigma", sigma=1)
                mu = sum( alpha[i] * x[:,i] for i in range(N))
                Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=y)
                posterior = pm.sample(progressbar=False, return_inferencedata=False)
            coeffs_ = posterior["alpha"][-1,:]
        else:
            reg = linear_model.LinearRegression()
            reg.fit(x, y)
            coeffs_ = reg.coef_
        
        self.coeffs = {key:coeffs_[i] for i,key in enumerate(self.vars)}
        return self.coeffs

    def extract_vars_from_matrix(self, A):
        vars = {}
        for i in range(self.natoms):
            for j in range(i+1, self.natoms):
                vars[i,j] = A[i,j]
        return vars

    def explore(self, explore_steps, verbose, **kwargs):
        fully_connected = kwargs.get("fully_connected", True)
        random = kwargs.get("random", True)
        valid = kwargs.get("valid", True)

        for step in range( explore_steps):
            A, graph = self.sample(random=True, valid=valid, fully_connected=fully_connected)
            target = self.target_func(self.x, A)
            vars = self.extract_vars_from_matrix(A)

            self.target_history.append( target )
            self.var_history.append( vars )
            self.graph_history.append( graph )
            self.mol_history.append( graph.mol )

            if verbose:
                print(f"Exploration step: {step}  Objective: {target}  Props: {graph.prop}", 
                        end="\r")
        self.explore_done = True

    def minimize(self, explore_steps=10, iterations = 25, verbose=True,**kwargs):
        fully_connected = kwargs.get("fully_connected", True)
        random = kwargs.get("random", True)
        valid = kwargs.get("valid", True)

        if not self.explore_done: self.explore(explore_steps, verbose, **kwargs)

        for step in range( iterations):
            self.update_coeffs()
            
            qubo_multiplier = np.max( list(self.coeffs.values()) )

            A, graph = self.sample( random = False, 
                                    valid=valid, 
                                    fully_connected=fully_connected,
                                    multiplier = qubo_multiplier,
                                    )

            target = self.target_func(self.x, A)
            vars = self.extract_vars_from_matrix(A)

            self.target_history.append( target )
            self.var_history.append( vars )
            self.graph_history.append( graph )
            self.mol_history.append( graph.mol )

            if verbose:
                print(f"Sampling step: {step}  Objective: {target}  Props: {graph.prop}",
                        end="\r")

        

    
def join(dic, key, value):
    try:
        dic[key] += value
    except KeyError:
        dic[key] = value