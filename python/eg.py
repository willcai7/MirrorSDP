from modules import *

C_generator = GaussianC(seed=224)
# C = C_generator.generate(10)
model = MaxCutModel(beta=10, dim=10, generator=C_generator)
model.solve(raw=False, verbose=True)