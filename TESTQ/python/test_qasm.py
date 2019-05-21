from qiskit.qasm import Qasm
qasm_string = """//this is a comment
OPENQASM 2.0;
qreg q[2]; // This is another comment
"""

qasm_string2 = """
// This is another comment
OPENQASM 2.0; // This is another comment
qreg q[2];// This is another comment

 
// This is another comment
qreg r[1];
// This is another comment
"""

node_circuit = Qasm(data=qasm_string).parse()
print(node_circuit.qasm())
print([ type(i) for i in node_circuit.children])

node_circuit = Qasm(data=qasm_string2).parse()
print(node_circuit.qasm())
print([ type(i) for i in node_circuit.children])