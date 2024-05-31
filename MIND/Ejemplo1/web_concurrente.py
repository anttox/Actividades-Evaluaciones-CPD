import asyncio # Programacion asincrona 

# Se define  una funcion asincronica para manejar las solicitudes entrantes
async def handle_request(reader, writer):
    # Se lee hasta 100 bytes de los datos del cliente
    data = await reader.read(100)
    # Decodificamos los datos recibidos a una cadena
    message = data.decode()
    # Imprimimos el mensaje recibido
    print(f"Received {message}")
    # Escribimos los datos recibidos de vuelta al cliente
    writer.write(data)
    # Esperamos hasta que los datos se hayan enviado completamente
    await writer.drain()
    # Cerramos la conexion con el cliente
    writer.close()

# Se define la funcion principal asincronica
async def main():
    # Iniciamos el servidor en la dirección 127.0.0.1 y el puerto 8888
    server = await asyncio.start_server(handle_request, '127.0.0.1', 8888)
    async with server:
        # Mantenemos el servidor en ejecución indefinidamente co server.serve_forever
        await server.serve_forever()

# Se ejecuta la función principal asincronica
asyncio.run(main())
