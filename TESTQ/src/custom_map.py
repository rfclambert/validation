"""
This module contains the definition of a base class for
feature map. Several types of commonly used approaches.
"""


import logging

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

from qiskit.aqua.components.feature_maps import FeatureMap
from inspect import signature

logger = logging.getLogger(__name__)


class CustomExpansion(FeatureMap):
    """
    Mapping data the way you want
    """

    CONFIGURATION = {
        'name': 'CustomExpansion',
        'description': 'Custom expansion for feature map (any order)',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'Custom_Expansion_schema',
            'type': 'object',
            'properties': {
                'feature_param': {
                    'type': ['array']
                }
            },
            'additionalProperties': False
        }
    }

    def __init__(self, num_qubits, constructor_function, feature_param):
        """Constructor.

        Args:
            num_qubits (int): number of qubits
            constructor_function (fun): a function that takes as parameters
            a datum x, a QuantumRegister qr, a boolean inverse and
            all other parameters needed from feature_param
            feature_param (list): the list of parameters needed to generate
            the circuit, that won't change depending on the data given
            (such as the data map function or other).
        """
        self.validate(locals())
        super().__init__()
        self._num_qubits = num_qubits
        self._feature_dimension = num_qubits
        sig = signature(constructor_function)
        if len(sig.parameters) != len(feature_param)+3:
            raise ValueError("The constructor_function given don't match the parameters given.\n" +
                             "Make sure it takes, in this order, the datum x, the QuantumRegister qr, the Boolean\n" +
                             " inverse and all the parameters provided in feature_param")
        self._constructor_function = constructor_function
        self._feature_param = feature_param

    def construct_circuit(self, x, qr=None, inverse=False):
        """
        Construct the circuit based on given data and according to the function provided at instantiation.

        Args:
            x (numpy.ndarray): 1-D to-be-transformed data.
            qr (QauntumRegister): the QuantumRegister object for the circuit, if None,
                                  generate new registers with name q.
            inverse (bool): whether or not inverse the circuit

        Returns:
            QuantumCircuit: a quantum circuit transform data x.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be numpy array.")
        if x.ndim != 1:
            raise ValueError("x must be 1-D array.")
        if x.shape[0] != self._num_qubits:
            raise ValueError("number of qubits and data dimension must be the same.")
        if qr is None:
            qr = QuantumRegister(self._num_qubits, name='q')
        qc = self._constructor_function(x, qr, inverse, *self._feature_param)
        return qc
