from tabnanny import check
import torch


def load_model(model, path):
    checkpoint = torch.load(path)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)

def check_isomer(mol1, mol2):
    return (mol1.x.numpy().sum(axis=0) == mol2.x.numpy().sum(axis=0)).prod()

# def CSR_to_sparse_tensor(X):
#     coo = X.tocoo()
#     indices = np.mat([coo.row, coo.col]).transpose()
#     return tf.SparseTensor(indices, coo.data, coo.shape)

# X = np.array([[ 0.    ,  1.    ,  0.    ,  0.    ,  0.    , -0.014 ,  1.1802, 0.0078,  0.    ,  0.    ],
#               [ 0.    ,  0.    ,  0.    ,  1.    ,  0.    ,  0.0023, -0.0197, 0.0022,  0.    ,  0.    ],
#               [ 1.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.915 ,  1.7895, 0.004 ,  0.    ,  0.    ],
#               [ 1.    ,  0.    ,  0.    ,  0.    ,  0.    , -0.9591,  1.764 , 0.0172,  0.    ,  0.    ]])

# Ai = csr_matrix(np.array([[0, 1, 1, 1],
#                         [1, 0, 0, 0],
#                         [1, 0, 0, 0],
#                         [1, 0, 0, 0]], dtype="int32"))
# A = CSR_to_sparse_tensor(Ai)

# E = np.array([[0., 1., 0., 0.],
#               [1., 0., 0., 0.],
#               [1., 0., 0., 0.],
#               [0., 1., 0., 0.],
#               [1., 0., 0., 0.],
#               [1., 0., 0., 0.]])

# I = np.zeros(X.shape[0], dtype="int32")

# class MPNN(keras.Model):
#         def __init__(self, embedding_size=256):
#             super().__init__()
#             self.ecc1 = spk.layers.ECCConv(5, activation="relu")
#             self.ecc2 = spk.layers.ECCConv(6, activation="relu")
#             #self.concat = NodeConcat()
#             self.globalpool = spk.layers.GlobalSumPool()
#             self.fc = keras.layers.Dense(embedding_size)
#         def call(self, inputs):
#             x, a, e, i = inputs
#             x1 = self.ecc1( [x,a,e] )
#             x2 = self.ecc2( [x1,a,e] )
#             #out1 = self.concat( [x2,a,e] )

#             x3 = self.globalpool( [x2, i] )
#             out2 = self.fc( x3 )
#             return out2 # loss function needed

# def generate_weights_for_testing(X,A,E,I, filename=None):

#     gnn = MPNN(256)
#     gnn_config = gnn_config = {"eccconv": {"layers": 2, 
#                           "channels": [5, 6],
#                           "activations": ["relu", "relu"]},
#                         "embedding": {"layers":1,
#                             "size": [256],
#                             "activations" : [None]}
#              }

#     reg_config = {}

#     gnn([X,A,E,I])
#     gnn_weights = gnn.get_weights()
#     gnn_weights[2] = np.random.random(size=gnn_weights[2].shape) * 2 - 1.0

#     gnn.set_weights(gnn_weights)

#     if filename is not None:
#         np.savez(filename, gnn_weights = gnn.get_weights(), reg_weights = [], rbm_weights = [],
#                 gnn_config = gnn_config, reg_config = reg_config)
    
#     return gnn
# #gnn_model = generate_weights_for_testing(X,A,E,I, "checkpoints\\conv_2r_1l.npz")
# #gnn_model = MPNN(256)
# #gnn_model([X,A,E,I])


# class Writer:
#     def __init__(self, name):
#         path = "logs/" + name + "/"
#         log_dir = path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#         self.writer = tf.summary.create_file_writer(log_dir)
#         print("tensorboard --logdir " + path  )
        
#     def log(self, data):
#         with self.writer.as_default():
#             for metric in data:
#                 step, value = data[metric]
#                 tf.summary.scalar(metric, value, step=step)
    
        
# class LoaderMod(DisjointLoader):
# #     def __init__(self, dataset, **kwargs):
# #         super().__init__(dataset, **kwargs)
#     def generator(self):
#         len_data = len(self.dataset)
#         batches = int(np.ceil(len_data / self.batch_size))
    
#     def batch_generator(self, data, batch_size, shuffle=True):
#         len_data = len(data[0])
#         batches = int(np.ceil(len_data / batch_size))
#         if shuffle:
#             shuffle_inplace(*data)
            


# def shuffle_inplace(*args):
#     rng_state = np.random.get_state()
#     for a in args:
#         np.random.set_state(rng_state)
#         np.random.shuffle(a)
