import asyncio

async def lazy_printer(delay, message):
    await asyncio.sleep(delay)
    print(message)
# Usar gather en vez de wait porque gather permite correr corutinas concurrentemente
# wait solo es ara esperar la finalizacion de las corutinas sin resultados
async def main():
    await asyncio.gather(
        lazy_printer(1, 'Lento'),
        lazy_printer(0, 'Full velocidad')
    )

asyncio.run(main())


