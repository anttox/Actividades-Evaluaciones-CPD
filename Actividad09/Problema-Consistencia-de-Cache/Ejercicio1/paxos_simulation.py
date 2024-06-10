import random

# Clase Proposer (Proponente)
class Proposer:
    def __init__(self, id, acceptors):
        self.id = id  # Identificador del proponente
        self.acceptors = acceptors  # Lista de aceptadores

    def propose(self, value):
        # Generamos un numero de propuesta aleatorio
        proposal_number = random.randint(1, 100)
        promises = 0  # Contador de promesas recibidas

        # Solicitante de  promesas de los aceptadores
        for acceptor in self.acceptors:
            if acceptor.promise(proposal_number):
                promises += 1

        # Si se reciben promesas de la mayoria se envia la propuesta para aceptacion
        if promises > len(self.acceptors) // 2:
            for acceptor in self.acceptors:
                acceptor.accept(proposal_number, value)
            return True  # Propuesta aceptada
        return False  # Propuesta rechazada

# Clase Acceptor (Aceptador)
class Acceptor:
    def __init__(self, id):
        self.id = id  # Identificador del aceptador
        self.promised_proposal = 0  # Numero de propuesta prometido
        self.accepted_proposal = 0  # Numero de propuesta aceptado
        self.accepted_value = None  # Valor aceptado

    def promise(self, proposal_number):
        # Prometemos no aceptar propuestas anteriores al numero de propuesta recibido
        if proposal_number > self.promised_proposal:
            self.promised_proposal = proposal_number
            return True
        return False

    def accept(self, proposal_number, value):
        # Aceptamos la propuesta si el numero de propuesta es mayor o igual al prometido
        if proposal_number >= self.promised_proposal:
            self.accepted_proposal = proposal_number
            self.accepted_value = value

# Ejemplo 
if __name__ == "__main__":
    # Creamos una lista de aceptadores
    acceptors = [Acceptor(i) for i in range(3)]
    # Creamos un proponente
    proposer = Proposer(1, acceptors)

    # Proponemos un valor y verificamos si es aceptado
    if proposer.propose("value1"):
        print("Proposal accepted")
    else:
        print("Proposal rejected")
