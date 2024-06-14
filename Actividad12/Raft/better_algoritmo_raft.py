import random
import threading
import time

# Definimos la clase Node que representa un nodo en el algoritmo de eleccion de lider Raft
class Node:
    def __init__(self, node_id):
        self.node_id = node_id
        self.state = "follower"
        self.votes = 0

    # Metodo para iniciar una eleccion
    def start_election(self):
        self.state = "candidate"
        self.votes = 1  # El nodo se vota a sí mismo
        print(f"Nodo {self.node_id} inicia una elección")

    # Metodo para recibir un voto
    def receive_vote(self):
        self.votes += 1
        print(f"Nodo {self.node_id} recibe un voto")

# Creacion de una lista de nodos
nodes = [Node(i) for i in range(5)]
leader_elected = threading.Event()

# Funcion que representa el comportamiento de un nodo
def run_node(node):
    while not leader_elected.is_set():
        if node.state == "follower" and random.random() < 0.2:  # Aumentar la probabilidad
            time.sleep(random.uniform(0.01, 0.05))  # Espera aleatoria antes de iniciar la elección
            node.start_election()
            for peer in nodes:
                if peer != node:
                    peer.receive_vote()
            # Si el nodo recibe la mayoría de los votos, se convierte en el líder
            if node.votes > len(nodes) / 2:
                node.state = "leader"
                leader_elected.set()
                print(f"Nodo {node.node_id} es elegido como líder")
            else:
                node.state = "follower"  # Resetear el estado si no gana la elección
                node.votes = 0  # Resetear los votos
        else:
            threading.Event().wait(0.05)  # Reducir el tiempo de espera entre iteraciones

# Creacion y ejecucion de hilos para cada nodo
threads = [threading.Thread(target=run_node, args=(node,)) for node in nodes]
for t in threads:
    t.start()
for t in threads:
    t.join()



