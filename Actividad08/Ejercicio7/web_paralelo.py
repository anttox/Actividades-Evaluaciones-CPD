import aiohttp # Peticiones HTTP de manera asincrona
import asyncio # Maneja la concurrencia

# Funcion para descargar el contenido de una pagina web y guardarlo en un archivo
# Session: realiza la solicitud
async def fetch_page(session, url):
    async with session.get(url) as response:
        # await para esperar a que se descargue el contenido de la pagina
        content = await response.text()
        # Genera un nombre de archivo a partir de la URL con url.replace
        filename = url.replace("https://", "").replace("/", "_") + ".html"
        # Escribimos el contenido en un archivo
        with open(filename, 'w') as file:
            file.write(content)
        return filename # Nombre del archivo en el que se guardo el contenido

# Descargamos varias paginas web en paralelo
async def parallel_download(urls):
    async with aiohttp.ClientSession() as session:
        # Creamos una lista de tareas para descargar cada pagina
        tasks = [fetch_page(session, url) for url in urls]
        # Se ejecuta las tareas en paralelo y espera a que todas terminen con asyncio.gather
        return await asyncio.gather(*tasks) # Retornamos la lista de nombres de archivos en los que se guardo el contenido

# Lista de urls para descargar
# urls = ["https://example.com", "https://example.org", "https://example.net"]
urls = ["https://eticapr.com/", "https://www.bancomundial.org/es/home", "https://gruporedes.net/"]
# Se ejecuta la descarga paralela de las páginas web con asyncio.run
results = asyncio.run(parallel_download(urls))

# Imprimimos los nombres de los archivos en los que se guardaron los contenidos 
print(results)
