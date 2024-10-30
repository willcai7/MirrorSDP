import numpy as np

class GaussianC:
    def __init__(self,  seed=224):
        self.seed = seed
        self.type = 'gaussian'
    
    def generate(self, dim):
        np.random.seed(self.seed)
        C = np.random.randn(dim, dim)
        C = C + C.T
        return C