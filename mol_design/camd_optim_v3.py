import sys
from pyomo.environ import *
import tensorflow as tf
import numpy as np
import spektral as spk
from tensorflow import keras
from spektral.layers.convolutional.message_passing import MessagePassing
from utils import *
from pyomo.util.infeasible import log_infeasible_constraints
import QM9_fixed

# Dataset params
dataset = QM9_fixed.QM9_mod(amount=10)
idx = 5
mol = dataset[idx]
X, Ai, E = mol.x, mol.a, mol.e
A = CSR_to_sparse_tensor(Ai)
I = np.zeros(X.shape[0], dtype="int32")
Natoms = X.shape[0]
print("Atoms - ", Natoms)
Nf = X.shape[1]
Ef = E.shape[1]

# load config
gnn_model = generate_weights_for_testing(X,A,E,I)
gnn_model([X,A,E,I])
# generate sample weights


def load_model_config(filename):
    data = np.load(filename, allow_pickle=True)
    return data["gnn_weights"], data["reg_weights"], data["rbm_weights"], data["gnn_config"].item(), data["reg_config"].item()

gnn, reg, rbm, gconf, rconf = load_model_config("checkpoints\\conv_2r_1l.npz")

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

model = ConcreteModel()

def add_mlp_vars(m):
    m.edges = Set(initialize = [(i,j) for i in range(Natoms) for j in range(Natoms) if i!=j])
    m.E = Var(m.edges, range(Ef), within=Binary, initialize=0)
    m.A = Var(m.edges, within=Binary, initialize=0)
    m.c = ConstraintList()
    for (i,j) in m.edges:
        m.c.add( sum(m.E[i,j,k] for k in range(Ef)) <= 1 )
        m.c.add( sum(m.E[i,j,k] for k in range(Ef)) == m.A[i,j] )
        m.c.add( m.A[i,j] == m.A[j,i] )
        for k in range(Ef):
            m.c.add( m.E[i,j,k] == m.E[j,i,k] )
    
    m.mlp_conv = Set(initialize = [(layer,i,j,feature) for layer in range(Nconv) for (i,j) in m.edges for feature in range(conv_features[layer]*conv_features[layer+1])])
    m.mlp = Var(m.mlp_conv)
    return m

model = add_mlp_vars(model)

def add_mlp_constraints(m):
    spacing = 3
    start = 1

    # not skipping constraints coz i !=j in mlp_conv set
    # def con1(m, layer, atom_i, atom_j, feature):
    #     Wm = gnn[spacing*layer + start]
    #     bm = gnn[spacing*layer + start + 1]
    #     if atom_i != atom_j:
    #         nn_value = sum( Wm[edge, feature] * m.E[atom_i, atom_j, edge] for edge in range(Ef)) + bm[feature]
    #         return nn_value == m.mlp[layer, atom_i, atom_j, feature]
    #     else:
    #         return Constraint.Skip
    # m.cm1 = Constraint(m.mlp_conv, rule=con1)

    def con1a(m, layer, atom_i, atom_j, feature):
        # Glovers linearization for MLP output 
        return -99999*m.A[atom_i, atom_j] <= m.mlp[layer,atom_i,atom_j,feature]
    m.cm1a = Constraint(m.mlp_conv, rule=con1a)

    def con1b(m, layer, atom_i, atom_j, feature):
        # Glovers linearization for MLP output - part 1b
        return  m.mlp[layer,atom_i,atom_j,feature] <= 99999*m.A[atom_i,atom_j]
    m.cm1b = Constraint(m.mlp_conv, rule=con1b)

    def con1c(m, layer, atom_i, atom_j, feature):
        # Glovers linearization for MLP output - part 2a
        Wm = gnn[spacing*layer + start]
        bm = gnn[spacing*layer + start + 1]
        nn_value = sum( Wm[edge, feature] * m.E[atom_i, atom_j, edge] for edge in range(Ef)) + bm[feature]
        return nn_value - 99999*(1-m.A[atom_i,atom_j]) <= m.mlp[layer,atom_i,atom_j,feature]
    m.cm1c = Constraint(m.mlp_conv, rule=con1c)

    def con1d(m, layer, atom_i, atom_j, feature):
        # Glovers linearization for MLP output - part 2b
        Wm = gnn[spacing*layer + start]
        bm = gnn[spacing*layer + start + 1]
        nn_value = sum( Wm[edge, feature] * m.E[atom_i, atom_j, edge] for edge in range(Ef)) + bm[feature]
        return m.mlp[layer,atom_i,atom_j,feature] <= nn_value + 99999*(1-m.A[atom_i,atom_j])
    m.cm1d = Constraint(m.mlp_conv, rule=con1d)
    
    return model

model = add_mlp_constraints(model)

def add_gnn_vars(m):
    # dont set domain to non-negative in case of No relus
    m.conv = Set(initialize = [(i,j,k) for i in range(Nconv) for j in range(Natoms) for k in range(size_conv[i])])
    m.h = Var(m.conv) 
    m.yh = Var(m.conv, within=Binary)

    m.emb = Set(initialize = [(i,j) for i in range(Nemb) for j in range(size_emb[i])])
    m.v = Var(m.emb)
    m.yv = Var(m.emb, within=Binary)
    return m

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

    def con3(m, layer, atom, channel):
        spacing = 3
        start = 0
        F1 = conv_features[layer]
        F2 = conv_features[layer+1]
        Wo = gnn[spacing*layer + start]
        if layer == 0:
            nn_value_1 = sum(Wo[feature, channel] * X[atom, feature] for feature in range(F1))      #channel*F1+feature   #feature*F2+channel
            #nn_value_2 = sum( sum(m.A[atom, atom_j] * X[atom_j, feature] * m.mlp[layer, atom, atom_j, feature*F2+channel] for feature in range(F1)) for atom_j in range(Natoms) if atom_j != atom ) 
            nn_value_2 = sum( sum( X[atom_j, feature] * m.mlp[layer, atom, atom_j, feature*F2+channel] for feature in range(F1)) for atom_j in range(Natoms) if atom_j != atom ) 
        else:
            nn_value_1 = sum(Wo[feature, channel] * m.h[layer-1, atom, feature] for feature in range(F1))
            #nn_value_2 = sum(m.A[atom, atom_j] * m.h[layer-1, atom_j, feature] * m.mlp[layer, atom, atom_j, feature*F2+channel] for feature in range(F1) for atom_j in range(Natoms) if atom_j != atom)
            nn_value_2 = sum( m.h[layer-1, atom_j, feature] * m.mlp[layer, atom, atom_j, feature*F2+channel] for feature in range(F1) for atom_j in range(Natoms) if atom_j != atom)

        if act_conv[layer] == "relu":
            return m.h[layer, atom, channel] >= nn_value_1 + nn_value_2
        else:
            return m.h[layer, atom, channel] == nn_value_1 + nn_value_2 # equality for linear
    m.cg3 = Constraint(m.conv, rule=con3)
    
    def con4(m, layer, atom, channel):
        if act_conv[layer] == "relu":
            return m.h[layer, atom, channel] <= 99999 * m.yh[layer, atom, channel]
        else:
            print("Skipping con4")
            return Constraint.Skip
    m.cg4 = Constraint(m.conv, rule=con4)

    def con5(m, layer, atom, channel):
        spacing = 3
        start = 0
        F1 = conv_features[layer]
        F2 = conv_features[layer+1]
        Wo = gnn[spacing*layer + start]
        if layer == 0:
            nn_value_1 = sum(Wo[feature, channel] * X[atom, feature] for feature in range(conv_features[layer]))
        #    nn_value_2 = sum(m.A[atom, atom_j] * X[atom_j, feature] * m.mlp[layer, atom, atom_j, feature*F2+channel] for feature in range(F1) for atom_j in range(Natoms) if atom_j != atom)
            nn_value_2 = sum( X[atom_j, feature] * m.mlp[layer, atom, atom_j, feature*F2+channel] for feature in range(F1) for atom_j in range(Natoms) if atom_j != atom)
        else:
            nn_value_1 = sum(Wo[feature, channel] * m.h[layer-1, atom, feature] for feature in range(conv_features[layer]))
        #    nn_value_2 = sum(m.A[atom, atom_j] * m.h[layer-1, atom_j, feature] * m.mlp[layer, atom, atom_j, feature*F2+channel] for feature in range(F1) for atom_j in range(Natoms) if atom_j != atom)
            nn_value_2 = sum( m.h[layer-1, atom_j, feature] * m.mlp[layer, atom, atom_j, feature*F2+channel] for feature in range(F1) for atom_j in range(Natoms) if atom_j != atom)
        if act_conv[layer] == "relu":
            return m.h[layer, atom, channel] <= nn_value_1 + nn_value_2 + 99999*(1-m.yh[layer, atom, channel])
        else:
            print("Skipping con5")
            return Constraint.Skip
    m.cg5 = Constraint(m.conv, rule=con5)

    return m

model = add_gnn_conv_constraints(model)

def add_gnn_emb_constraints(m):
    # Emb layers
    def global_sum_pool(m, channel): # operation required for generating embedding
        return sum( m.h[Nconv-1, atom, channel] for atom in range(Natoms) )
    
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
        start = Nconv * 3
        F1 = emb_features[layer]
        if layer == 0:
            nn_value_1 = sum(global_sum_pool(m, channel) * gnn[spacing*layer+start][channel, neuron] for channel in range(F1)) 
            nn_value_2 = gnn[spacing*layer+start+1][neuron]
        else:
            nn_value_1 = sum(m.v[layer-1, channel] * gnn[spacing*layer+start][channel, neuron] for channel in range(F1)) 
            nn_value_2 = gnn[spacing*layer+start+1][neuron]
        
        if act_conv[layer] == "relu":
            return m.v[layer, neuron] >= nn_value_1 + nn_value_2
        else:
            return m.v[layer, neuron] == nn_value_1 + nn_value_2
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
        start = Nconv * 3
        F1 = emb_features[layer]
        if layer == 0:
            nn_value_1 = sum(global_sum_pool(m, channel) * gnn[spacing*layer+start][channel, neuron] for channel in range(F1)) 
            nn_value_2 = gnn[spacing*layer+start+1][neuron]
        else:
            nn_value_1 = sum(m.v[layer-1, channel] * gnn[spacing*layer+start][channel, neuron] for channel in range(F1)) 
            nn_value_2 = gnn[spacing*layer+start+1][neuron]
        
        if act_conv[layer] == "relu":
            return m.v[layer, neuron] <= nn_value_1 + nn_value_2 + 99999*(1-m.yv[layer,neuron])
        else:
            print("Skipping con3")
            return Constraint.Skip
    m.ce5 = Constraint(m.emb, rule=con5)

    return model

model = add_gnn_emb_constraints(model)

def add_free_energy_objective(m):
    def obj_rule(m):
        hidden_term = sum( np.log(1 + np.exp(sum(rbm.W[vis, hid] * m.v[Nemb-1, vis] for vis in range(emb_features[-1])) + rbm.c[hid])) for hid in range(rbm.c.shape[0]) )
        visible_term = - sum( m.v[Nemb-1, vis] * rbm.b[vis] for vis in range(emb_features[-1]) )
        return visible_term - hidden_term
    model.obj = Objective(rule=obj_rule, sense=minimize)

    return model

model = add_free_energy_objective(model)
#model.obj = Objective(expr = 0)

def test_model(solver_name):
    #output = gnn_model([X,A,E,I])
    gnn_model.set_weights(gnn)

    #obs2 = gnn_model.ecc1.kernel_network_layers[0](E).numpy()
    obs = gnn_model([X,A,E,I]).numpy()
    #layer = 1
    # for atom in range(Natoms):
    #     for channel in range(conv_features[layer+1]):
    #         model.h[layer, atom, channel].fix( obs[atom, channel] )
    
    emb_layer = 0
    for neuron in range(emb_features[emb_layer+1]):
        model.v[emb_layer, neuron].fix( obs[0,neuron] )

    # count = 0
    # for i in range(Natoms):
    #     for j in range(i+1, Natoms):
    #         if Ai[i,j] == 1:
    #             model.A[i,j].fix(1)
    #             model.A[j,i].fix(1)
    #             for e in range(Ef):
    #                 if E[count,e] == 1:
    #                     model.E[i,j,e].fix(1)
    #                     model.E[j,i,e].fix(1)
    #                 else:
    #                     model.E[i,j,e].fix(0)
    #                     model.E[j,i,e].fix(0)
    #             count += 2 
    #         else:
    #             model.A[i,j].fix(0)
    #             model.A[j,i].fix(0)
    #             for e in range(Ef):
    #                 model.E[i,j,e].fix(0)
    #                 model.E[j,i,e].fix(0)

    #eps = 0.1
    #def relaxed_bounds(m, atom, channel):
    #    return (obs[atom, channel] - eps, m.h[layer, atom, channel], obs[atom, channel] + eps)
    #model.bb = Constraint(range(Natoms), range(conv_features[layer+1]), rule=relaxed_bounds)

    solver = SolverFactory(solver_name)
    solver.options['NonConvex'] = 2
    solver.solve(model, tee=True)

    #log_infeasible_constraints(model, log_expression=True)    
    Ap = np.zeros((Natoms, Natoms))
    #Ep = []
    Epf = np.zeros((Natoms, Natoms, Ef))

    for i in range(Natoms):
        for j in range(i+1, Natoms):
            Ap[i,j] = model.A[i,j]()
            Ap[j,i] = model.A[j,i]()
            Epf[i,j,:] = model.E[i,j,:]() 
            Epf[j,i,:] = model.E[j,i,:]()

    # Hp = np.zeros((Natoms,conv_features[layer+1]))
    # for (layer,i,channel) in model.conv:
    #     Hp[i,channel] = model.h[layer,i,channel]()

    # mlp = []
    # mlpF = np.zeros((Natoms, Natoms, 50))
    # for i in range(Natoms):
    #     for j in range(i+1,Natoms):
    #         if Ap[i,j] == 1: # consecutive edges are same
    #             mlp.append( model.mlp[layer,i,j,:]() )
    #             mlp.append( model.mlp[layer,j,i,:]() )

    # for i in range(Natoms):
    #     for j in range(Natoms):
    #         for f in range(50):
    #             try:
    #                 mlpF[i,j,f] = model.mlp[layer,i,j,f]()
    #             except:
    #                 pass
                

    return model, Ap, Epf#, Hp, np.array(mlp), mlpF

model, Ap, Epf = test_model(sys.argv[1])

def nn_output():
    mlpN = np.zeros((Natoms, Natoms, 50))
    for i in range(Natoms):
        for j in range(i+1, Natoms):
            Oij = gnn_model.ecc1.kernel_network_layers[0](Epf[i,j,:].reshape(1,-1)).numpy()
            Oji = gnn_model.ecc1.kernel_network_layers[0](Epf[j,i,:].reshape(1,-1)).numpy()
            mlpN[i,j,:] = Oij
            mlpN[j,i,:] = Oji

    Hnc = np.zeros((Natoms, conv_features[1]))

    layer = 0
    Wo = gnn[0]
    F1 = conv_features[0]
    F2 = conv_features[1]

    # atom = 1
    # channel = 0
    #     #for channel in range(conv_features[1]):
    # nn_value_1 = sum(Wo[feature, channel] * X[atom, feature] for feature in range(F1))
    # print("NN value 1 : ", nn_value_1)
    # nn_value_2 = 0
    # for atom_j in range(Natoms):
    #     if atom_j != atom and Ap[atom, atom_j] == 1:
    #         print("N : ", atom_j)
    #         value = sum( X[atom_j, feature] * mlpN[atom, atom_j, feature*F2+channel] for feature in range(F1))  #feature*F2+channel
    #         print("Value : ", value)
    #         nn_value_2 += value
    # print("NN Val 2 : ", nn_value_2)
    # Hnc[atom, channel] = nn_value_1 + nn_value_2
    
    Hn = gnn_model([X,A,E,I])

    Hnc0 = np.zeros((Natoms, conv_features[1]))
    Hnc1 = np.zeros((Natoms, conv_features[1]))

    for atom in range(Natoms):
        for channel in range(conv_features[1]):
            Hnc0[atom,channel] = sum(Wo[feature, channel] * X[atom, feature] for feature in range(F1))
            for atom_j in range(Natoms):
                if atom_j != atom and Ap[atom, atom_j] == 1:
                    value = sum( X[atom_j, feature] * mlpN[atom, atom_j, feature*F2+channel] for feature in range(F1))  #feature*F2+channel
                    Hnc1[atom, channel] += value

    Hnc = Hnc0 + Hnc1

    return mlpN, Hn, Hnc

#mlpn, Hn, Hnc = nn_output()
