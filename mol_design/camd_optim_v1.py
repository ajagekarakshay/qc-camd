from pyomo.environ import *
import tensorflow as tf
import numpy as np
import spektral as spk
from tensorflow import keras
from spektral.layers.convolutional.message_passing import MessagePassing
from utils import *
from pyomo.util.infeasible import log_infeasible_constraints

# Dataset params
dataset = spk.datasets.QM9(amount=10)
idx = 5
mol = dataset[idx]
X, A, E = mol.x, mol.a, mol.e
A = CSR_to_sparse_tensor(A)
I = np.zeros(X.shape[0], dtype="int32")
Natoms = X.shape[0]
Nf = X.shape[1]
Ef = E.shape[1]

# generate sample weights
gnn_model = generate_weights_for_testing(X,A,E,I, "checkpoints\\conv_linear_1.npz")

def load_model_config(filename):
    data = np.load(filename, allow_pickle=True)
    return data["gnn_weights"], data["reg_weights"], data["rbm_weights"], data["gnn_config"].item(), data["reg_config"].item()

gnn, reg, rbm, gconf, rconf = load_model_config("checkpoints\\conv_linear_1.npz")

# Conv params
Nconv = gconf['eccconv']['layers']
size_conv = gconf['eccconv']['channels']
conv_features = [Nf] # to use feautres at each layer including input
conv_features.extend(size_conv)
act_conv = gconf['eccconv']['activations']

Nemb = gconf['embedding']['layers']
size_emb = gconf['embedding']['size']
emb_features = [conv_features[-1]]
emb_features.extend(size_emb)
act_emb = gconf['embedding']['activations']

# optimization model

model = ConcreteModel()

def add_gnn_vars(m):
    # dont set domain to non-negative in case of No relus
    m.conv = Set(initialize = [(i,j,k) for i in range(Nconv) for j in range(Natoms) for k in range(size_conv[i])])
    m.h = Var(m.conv) 
    m.yh = Var(m.conv, within=Binary)

    m.emb = Set(initialize = [(i,j) for i in range(Nemb) for j in range(size_emb[i])])
    m.v = Var(m.emb)
    m.yv = Var(m.emb, within=Binary)

    # Edge variables and linking constraints (maybe add Aij = Aji later)
    m.edges = Set(initialize = [(i,j) for i in range(Natoms) for j in range(Natoms) if i!=j])
    m.e = Var(m.edges, range(Ef), within=Binary)
    m.a = Var(m.edges, within=Binary)
    m.c = ConstraintList()
    # for i in range(Natoms):
    #     for j in range(Natoms):
    #         if i!=j:
    #             m.c.add( sum(m.e[i,j,k] for k in range(Ef)) <= 1 )
    #             m.c.add( sum(m.e[i,j,k] for k in range(Ef)) == m.a[i,j] )
    return model

model = add_gnn_vars(model)

def add_gnn_conv_constraints(m):
    # conv layers
    def con1(m, layer, atom, channel):
        if act_conv[layer] == "relu":
            return m.h[layer, atom, channel] >= 0
        else:
            print("Skipping con1")
            return Constraint.Skip
    m.cg1 = Constraint(m.conv, rule=con1)

    def con2(m, layer, atom, channel):
        if act_conv[layer] == "relu":
            return m.h[layer, atom, channel] <= 99999
        else:
            print("Skipping con2")
            return Constraint.Skip
    m.cg2 = Constraint(m.conv, rule=con2)

    m.mlp_conv = Set(initialize = [(l,i,j,f) for l in range(Nconv) for (i,j) in m.edges for f in range(conv_features[l+1])])
    m.mlp = Var(m.mlp_conv) # no activation for MLP
    

    # def con3a(m, layer, atom_i, atom_j, channel):
    #     # Glovers linearization for MLP output - part 1a
    #     if atom_i != atom_j:
    #         return -99999*m.a[atom_i, atom_j] <= m.mlp[layer,atom_i,atom_j,channel]
    #     else:
    #         return Constraint.Skip
    # m.cg3a = Constraint(m.mlp_conv, rule=con3a)

    # def con3b(m, layer, atom_i, atom_j, channel):
    #     # Glovers linearization for MLP output - part 1b
    #     if atom_i != atom_j:
    #         return  m.mlp[layer,atom_i,atom_j,channel] <= 99999*m.a[atom_i,atom_j]
    #     else:
    #         return Constraint.Skip
    # m.cg3b = Constraint(m.mlp_conv, rule=con3b)

    def con3c(m, layer, atom_i, atom_j, channel):
        # Glovers linearization for MLP output - part 2a
        spacing = 3
        start = 1
        if atom_i != atom_j:
            F1 = conv_features[layer]
            if layer == 0:
                Oij = sum( X[atom_j,feature] * (sum(gnn[spacing*layer+start][e,channel*F1+feature] * m.e[atom_i, atom_j, e] for e in range(Ef)) + gnn[spacing*layer+start+1][channel*F1+feature]) for feature in range(F1) )
            else:
                Oij = sum( m.h[layer-1,atom_j,feature] * (sum(gnn[spacing*layer+start][e,channel*F1+feature] * m.e[atom_i, atom_j, e] for e in range(Ef)) + gnn[spacing*layer+start+1][channel*F1+feature]) for feature in range(F1) )
            return m.mlp[layer, atom_i, atom_j, channel] == Oij * m.a[atom_i, atom_j]
            #Oij - 99999*(1-m.a[atom_i, atom_j]) <= m.mlp[layer, atom_i, atom_j, channel] # changed (PLEASEEEEEEEEEEEEEEEEE check)
        else:
            return Constraint.Skip
    m.cg3c = Constraint(m.mlp_conv, rule=con3c)

    # def con3d(m, layer, atom_i, atom_j, channel):
    #     # Glovers linearization for MLP output - part 2b
    #     spacing = 3
    #     start = 1
    #     if atom_i != atom_j:
    #         F1 = conv_features[layer]
    #         if layer == 0:
    #             Oij = sum( X[atom_j,feature] * (sum(gnn[spacing*layer+start][e,channel*F1+feature] * m.e[atom_i, atom_j, e] for e in range(Ef)) + gnn[spacing*layer+start+1][channel*F1+feature]) for feature in range(F1) )
    #         else:
    #             Oij = sum( m.h[layer-1,atom_j,feature] * (sum(gnn[spacing*layer+start][e,channel*F1+feature] * m.e[atom_i, atom_j, e] for e in range(Ef)) + gnn[spacing*layer+start+1][channel*F1+feature]) for feature in range(F1) )
    #         return  m.mlp[layer, atom_i, atom_j, channel] <= Oij + 99999*(1-m.a[atom_i, atom_j])
    #     else:
    #         return Constraint.Skip
    # m.cg3d = Constraint(m.mlp_conv, rule=con3d)
    
    # def con3e(m, layer, atom, channel):
    #     # NN value output (Added both RELU and non-RELU constraints)
    #     spacing = 3
    #     start = 0
    #     if layer == 0:
    #         nn_value = sum(gnn[spacing*layer+start][feature,channel] * X[atom,feature] for feature in range(conv_features[layer])) + sum(m.mlp[layer,atom,atom_j,channel] for atom_j in range(Natoms) if atom_j!=atom)
    #     else:
    #         nn_value = sum(gnn[spacing*layer+start][feature,channel] * m.h[layer-1,atom,feature] for feature in range(conv_features[layer])) + sum(m.mlp[layer,atom,atom_j,channel] for atom_j in range(Natoms) if atom_j!=atom)
        
    #     if act_conv[layer] == "relu":
    #         return m.h[layer, atom, channel] >= nn_value
    #     else:
    #         return m.h[layer, atom, channel] == nn_value
    # m.cg3e = Constraint(m.conv, rule=con3e)
    
    def con3(m, layer, atom, channel):
        spacing = 3
        start = 0
        if layer == 0:
            nn_value = sum(gnn[spacing*layer+start][feature,channel] * X[atom,feature] for feature in range(conv_features[layer])) + sum(m.mlp[layer,atom,atom_j,channel] for atom_j in range(Natoms) if atom_j!=atom)
        else:
            nn_value = sum(gnn[spacing*layer+start][feature,channel] * m.h[layer-1,atom,feature] for feature in range(conv_features[layer])) + sum(m.mlp[layer,atom,atom_j,channel] for atom_j in range(Natoms) if atom_j!=atom)
        # linear
        return m.h[layer, atom, channel] == nn_value
    m.cg3 = Constraint(m.conv, rule=con3)
    
    def con4(m, layer, atom, channel):
        if act_conv[layer] == "relu":
            return m.h[layer, atom, channel] <= 99999 * m.yh[layer, atom, channel]
        else:
            print("Skipping con4")
            return Constraint.Skip
    m.cg4 = Constraint(m.conv, rule=con4)

    def con5(m, layer, atom, channel):
        # NN value output + M
        spacing = 3
        start = 0
        if act_conv[layer] == "relu":
            if layer == 0:
                nn_value = sum(gnn[spacing*layer+start][feature,channel] * X[atom,feature] for feature in range(conv_features[layer])) + sum(m.mlp[layer,atom,atom_j,channel] for atom_j in range(Natoms) if atom_j!=atom)
            else:
                nn_value = sum(gnn[spacing*layer+start][feature,channel] * m.h[layer,atom,feature] for feature in range(conv_features[layer])) + sum(m.mlp[layer,atom,atom_j,channel] for atom_j in range(Natoms) if atom_j!=atom)
            return m.h[layer, atom, channel] <= nn_value + 99999*(1-m.yh[layer, atom, channel])
        else:
            print("Skipping con5")
            return Constraint.Skip
    m.cg5 = Constraint(m.conv, rule=con5)
    return m

model = add_gnn_conv_constraints(model)

def add_gnn_emb_constraints(m):
    # Emb layers
    def global_sum_pool(m, channel): # operation required for generating embedding
        return sum(m.h[Nconv-1, atom, channel] for atom in range(Natoms)) # for final layer only

    def con1(m, layer, neuron):
        if act_emb[layer] == "relu":
            return m.v[layer, neuron] >= 0
        else:
            print("Skipping con1")
            return Constraint.Skip
    m.ce1 = Constraint(m.emb, rule=con1)

    def con2(m, layer, neuron):
        if act_emb[layer] == "relu":
            return m.v[layer, neuron] <= 99999
        else:
            print("Skipping con2")
            return Constraint.Skip
    m.ce2 = Constraint(m.emb, rule=con2)

    def con3(m, layer, neuron):
        spacing = 2
        start = 6
        F1 = emb_features[layer]
        if layer == 0:
            nn_value = sum(global_sum_pool(m, channel) * gnn[spacing*layer+start][channel, neuron] for channel in range(F1)) + gnn[spacing*layer+start+1][neuron]
        else:
            nn_value = sum(m.v[layer-1, channel] * gnn[spacing*layer+start][channel, neuron] for channel in range(F1)) + gnn[spacing*layer+start+1][neuron]
        
        if act_conv[layer] == "relu":
            return m.v[layer, neuron] >= nn_value
        else:
            print("Skipping con3")
            return m.v[layer, neuron] == nn_value
    m.ce3 = Constraint(m.emb, rule=con3)

    def con4(m, layer, neuron):
        if act_emb[layer] == "relu":
            return m.v[layer, neuron] <= 99999 * m.yv[layer,neuron]
        else:
            print("Skipping con4")
            return Constraint.Skip
    m.ce4 = Constraint(m.emb, rule=con4)

    def con5(m, layer, neuron):
        spacing = 2
        start = 6
        F1 = emb_features[layer]
        if layer == 0:
            nn_value = sum(global_sum_pool(m, channel) * gnn[spacing*layer+start][channel, neuron] for channel in range(F1)) + gnn[spacing*layer+start+1][neuron]
        else:
            nn_value = sum(m.v[layer-1, channel] * gnn[spacing*layer+start][channel, neuron] for channel in range(F1)) + gnn[spacing*layer+start+1][neuron]

        if act_conv[layer] == "relu":
            return m.v[layer, neuron] <= nn_value + 99999*(1-m.yv[layer,neuron])
        else:
            print("Skipping con5")
            return Constraint.Skip
    m.ce5 = Constraint(m.emb, rule=con5)

    return model

#model = add_gnn_emb_constraints(model)

def add_objective(m):
    def obj_rule(m):
        return 0
    m.obj = Objective(rule=obj_rule)
    return m

model = add_objective(model)

def test_model(solver_name):
    output = gnn_model([X,A,E,I])
    gnn_model.set_weights(gnn)

    observed = gnn_model([X,A,E,I]).numpy()
    print(observed.shape)
    for atom in range(Natoms):
        for channel in range(conv_features[Nconv]):
            model.h[Nconv-1, atom, channel].fix( observed[atom, channel] )
    
    solver = SolverFactory(solver_name)
    solver.solve(model, tee=True)

    #log_infeasible_constraints(model, log_expression=True)    
    # Ap = np.zeros((Natoms, Natoms))
    # Ep = []

    # for i,j in model.edges:
    #     Ap[i,j] = model.a[i,j]()
    
    # return Ap

test_model("baron")