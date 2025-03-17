from abc import ABC, abstractmethod


class BaseModelBuilder(ABC):
    def __init__(self):

        self.learning_rate_min = 1e-6
        self.learning_rate_max = 0.01

        
    @abstractmethod
    def build_model(self, hp):
        """
        Méthode abstraite devant retourner un modèle compilé,
        en utilisant l'objet 'hp' pour définir les hyperparamètres.
        """
        raise NotImplementedError("Please Implement this method")
    
    # @abstractmethod
    # def block(self, x, block_id):
    #     """
    #     Build a dense block with a tunable number of layers.
        
    #     For each layer index in 0...self.max_layers-1, we use self.hp.conditional_scope so that
    #     the hyperparameters (units, activation, dropout, etc.) are only registered if the
    #     current layer index is within the active number of layers.
        
    #     Parameters:
    #         x: Input tensor.
    #         block_id: Identifier string for this block (used for hyperparameter naming).
            
    #     Returns:
    #         The output tensor after applying the dense block.
    #     """
    #     raise NotImplementedError("Please Implement this method")
    
    
        
    
    
    
    
    
    