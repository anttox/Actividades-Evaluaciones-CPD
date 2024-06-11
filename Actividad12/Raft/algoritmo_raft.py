import random
import threading

class Node:
    def __init__(self, node_id):
        # Inicializa el nodo con su ID y establece su estado inicial como "follower"
        self.node_id = node_id
        self.state = "follower"
        self.votes = 0

    def start_election(self):
        # Cambia el estado del nodo a "candidate" y vota por sí mismo
        self.state = "candidate"
        self.votes = 1  # Vota por sí mismo
        print(f"Nodo {self.node_id} inicia una elección")

    def receive_vote(self):
        # Incrementa el número de votos recibidos por el nodo
        self.votes += 1
        print(f"Nodo {self.node_id} recibe un voto")

# Crea una lista de nodos
nodes = [Node(i) for i in range(5)]
# Evento para señalar la elección de un líder
leader_elected = threading.Event()

def run_node(node):
    # Función que ejecuta el comportamiento de un nodo
    while not leader_elected.is_set():
        if node.state == "follower" and random.random() < 0.05:
            # Inicia una elección con una probabilidad del 5%
            node.start_election()
            for peer in nodes:
                if peer != node:
                    peer.receive_vote()
            if node.votes > len(nodes) / 2:
                # Si el nodo recibe más de la mitad de los votos, se convierte en líder
                node.state = "leader"
                leader_elected.set()
                print(f"Nodo {node.node_id} es elegido como líder")

# Crear y arrancar hilos para cada nodo
threads = [threading.Thread(target=run_node, args=(node,)) for node in nodes]
for t in threads:
    t.start()
for t in threads:
    t.join()

