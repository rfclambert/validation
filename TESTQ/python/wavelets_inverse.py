# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.aqua.components.qfts import QFT
from wavelets_utils import *


class IWavelets(QFT):
    """A Wavelet Transform."""

    CONFIGURATION = {
        'name': 'IWAVELETS',
        'description': 'Inverse QWT',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'std_iqft_schema',
            'type': 'object',
            'properties': {
            },
            'additionalProperties': False
        }
    }

    def __init__(self, num_qubits):
        super().__init__()
        self._num_qubits = num_qubits

    def _build_matrix(self):
        return wave_coefs(2**self._num_qubits).transpose()

    def _build_circuit(self, qubits=None, circuit=None, do_swaps=True):
        return self.construct_circuit('circuit', qubits, circuit, True)

    def construct_circuit(self, mode='circuit', register=None, circuit=None, do_swaps=True):
        if mode == 'vector':
            return wave_coefs(2**self._num_qubits)
        elif mode == 'circuit':
            if register is None:
                register = QuantumRegister(self._num_qubits, name='q')
            if circuit is None:
                circuit = QuantumCircuit(register)
                CZeroP(circuit, register, register[0])
                Qn(circuit, register, [register[i] for i in range(self._num_qubits)])
                COne(circuit, register, register[0])
            return circuit.inverse()
        else:
            raise ValueError('Mode should be either "vector" or "circuit"')
