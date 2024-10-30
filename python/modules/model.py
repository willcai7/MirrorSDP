import numpy as np 
import os 
from .data import *
import logging
import time 
import wandb 
import scipy as sp 


class MaxCutModel:
    def __init__(self, beta, dim, generator, b=None):
        self.beta = beta
        self.C = generator.generate(dim)
        self.C_type = generator.type
        self.dim = dim
        self.job_name = 'maxcut'
        if b is None:
            self.b = np.ones(self.dim)/self.dim
        else:
            self.b = b
    
    def solve(self, stepsize=None, max_iter=None, verbose=False, raw=False, log_gap=20):
        if stepsize is None:
            stepsize = 2/(self.beta)
        if max_iter is None:
            max_iter = np.ceil(100*self.beta)
        
        # Initialize the logger
        lambda_history = np.zeros((max_iter, self.dim))
        average_lambda_history = np.zeros((max_iter, self.dim))
        dual_objective_history = np.zeros(max_iter) 
        feasibility_history = np.zeros(max_iter)
        
        if verbose:
            logger = logging.getLogger() # logger object
            logger.setLevel(logging.DEBUG) # set the logging level
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') # format of the log message
            console_handler = logging.StreamHandler() # console handler
            console_handler.setLevel(logging.INFO) # set the logging level
            console_handler.setFormatter(formatter) # set the formatter
            logger.addHandler(console_handler) # add the console handler to the logger

        if not raw:
            timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
            directory = os.path.join('output',self.job_name,self.C_type, timestamp)
            os.makedirs(directory, exist_ok=True)
            log_file_path = os.path.join(directory, 'train.log')
            wandb_project = 'maxcut'
            wandb.init(project=wandb_project, name=timestamp +"_" +self.job_name + "_" + self.C_type)
            wandb.config.update({'beta': self.beta, 'stepsize': stepsize, 'max_iter': max_iter})
            file_handler = logging.FileHandler(log_file_path, mode="a") # file handler
            file_handler.setLevel(logging.DEBUG) # set the logging level
            file_handler.setFormatter(formatter) # set the formatter
            logger.addHandler(file_handler)


        # Initialize the dual variable
        lambda_ = np.zeros(self.dim) 


        # Main loop 
        for t in range(max_iter):

            # Compute the dual objective and gradient
            A_ = -self.C + np.diag(lambda_)
            A_ = sp.linalg.expm(self.beta*A_)
            traceA = np.trace(A_)
            # print(traceA)
            dual_obj = - self.b@lambda_ + traceA/self.beta
            gradient = self.b - np.diag(A_)/traceA 

            # Update the dual variable
            lambda_ = lambda_ + stepsize*np.linalg.norm(gradient,1)*np.sign(gradient)
            lambda_ = lambda_ - np.mean(lambda_) 

            # Log the history
            lambda_history[t] = lambda_
            average_lambda_history[t] = np.mean(lambda_history[np.floor(t/2).astype(int):t+1], axis=0)
            dual_objective_history[t] = dual_obj
            feasibility_history[t] = np.linalg.norm(gradient,1)
        
            # Print the progress
            if verbose and t % log_gap == 0:
                logger.info(f'Iter {t}, dual objective: {dual_obj}, feasibility: {np.linalg.norm(gradient,1)}')
                if not raw:
                    wandb.log({'iter':t, 'dual_objective': dual_obj, 'feasibility': np.linalg.norm(gradient,1)})
        
        # save the output and parameters
        if not raw:
            np.savez(os.path.join(directory, 'parameter.npz'), beta=self.beta, C=self.C, b=self.b, stepsize=stepsize, max_iter=max_iter)
            np.savez(os.path.join(directory, 'output.npz'), lambda_history=lambda_history, average_lambda_history=average_lambda_history, dual_objective_history=dual_objective_history, feasibility_history=feasibility_history)
            wandb.finish()


