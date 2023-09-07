from typing import Protocol

class RegularizationMethod(Protocol):
    def regularize_data(self):
        pass

class Lasso:
    def regularize_data(self):
        pass

class Ridge:
    def regularize_data(self):
        pass

class ElasticNet:
    def regularize_data(self):
        pass