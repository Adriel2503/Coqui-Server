"""
Servidor WebSocket para síntesis de voz XTTS v2 con streaming en tiempo real.
Proporciona síntesis de voz multilingüe con latencia <200ms.
"""

import json
import logging
import time
import torch
import uvicorn
import numpy as np
import soxr
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Variable global para almacenar el modelo (se carga una sola vez)
modelo = None

@asynccontextmanager
async def ciclo_vida_aplicacion(app: FastAPI):
    """
    Gestor del ciclo de vida de la aplicación.
    Carga el modelo al iniciar y libera recursos al cerrar.
    """
    global modelo

    logger.info("Iniciando servidor...")

    try:
        # Rutas del modelo XTTS v2 (hardcodeadas)
        ruta_modelo = "/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2"
        ruta_config = "/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/config.json"

        logger.info(f"Cargando modelo desde: {ruta_modelo}")
        logger.info("Inicializando modelo...")
        config = XttsConfig()
        config.load_json(ruta_config)
        modelo = Xtts.init_from_config(config)
        modelo.load_checkpoint(config, checkpoint_dir=ruta_modelo, eval=True)

        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        modelo.to(dispositivo)
        logger.info(f"Modelo cargado en {dispositivo.upper()}")

    except Exception as e:
        logger.error(f"Error al cargar modelo: {e}", exc_info=True)
        raise

    yield

    # Limpiar recursos
    logger.info("Limpiando recursos...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="Servidor de Síntesis de Voz WebSocket",
    description="XTTS v2 con streaming en tiempo real",
    lifespan=ciclo_vida_aplicacion
)


# ============= FUNCIONES AUXILIARES =============

def validar_solicitud(solicitud: dict) -> tuple[str, str, str]:
    """
    Valida y extrae parámetros de la solicitud.

    Returns:
        tuple: (texto, idioma, archivo_speaker)
    """
    texto = solicitud.get("texto", "").strip()
    idioma = solicitud.get("idioma", "es").lower()
    archivo_speaker = solicitud.get("archivo_speaker", "hermana_carlos_mono.wav")

    logger.info(f"Parámetros: texto={len(texto)} chars, idioma={idioma}, speaker={archivo_speaker}")

    return texto, idioma, archivo_speaker


def procesar_chunk_audio(chunk: torch.Tensor) -> bytes:
    """
    Procesa un chunk de audio: resampling y conversión a bytes.

    Args:
        chunk: Tensor de audio en 24kHz float32

    Returns:
        bytes: Audio procesado en 8kHz int16 PCM
    """
    # Convertir de float32 24kHz a int16 8kHz PCM
    audio_np = chunk.cpu().numpy()
    # Resamplear de 24kHz a 8kHz con SOXR (75% más rápido que librosa)
    audio_8k = soxr.resample(audio_np, 24000, 8000, quality='HQ')
    # Normalizar y convertir a int16
    audio_s16 = np.clip(audio_8k * 32767, -32768, 32767).astype(np.int16)
    # Convertir a bytes
    return audio_s16.tobytes()


async def enviar_chunk(websocket: WebSocket, audio_bytes: bytes, numero_chunk: int) -> None:
    """
    Envía un chunk de audio al cliente vía WebSocket.

    Args:
        websocket: Conexión WebSocket
        audio_bytes: Bytes del audio procesado
        numero_chunk: Número del chunk actual
    """
    # Enviar metadata del chunk en JSON
    chunk_metadata = {
        "tipo": "chunk",
        "numero": numero_chunk,
        "tamaño_bytes": len(audio_bytes),
        "es_final": False
    }
    await websocket.send_json(chunk_metadata)

    # Enviar los bytes puros del audio
    await websocket.send_bytes(audio_bytes)


# ============= ENDPOINTS =============

@app.get("/health")
async def verificar_salud():
    """Verificar estado del servidor."""
    return {
        "estado": "ok",
        "dispositivo": "cuda" if torch.cuda.is_available() else "cpu"
    }

@app.websocket("/ws/tts")
async def endpoint_websocket(websocket: WebSocket):
    """Endpoint WebSocket para síntesis de voz en streaming."""
    await websocket.accept()

    id_cliente = id(websocket)

    logger.info(f"Cliente conectado: {id_cliente}")

    try:
        while True:
            logger.info("Esperando solicitud del cliente...")
            datos = await websocket.receive_text()
            logger.info(f"Solicitud recibida: {len(datos)} bytes")

            solicitud = json.loads(datos)
            logger.info(f"JSON decodificado: {solicitud}")

            texto, idioma, archivo_speaker = validar_solicitud(solicitud)

            if not texto:
                logger.warning("Texto vacío recibido")
                await websocket.send_json({"error": "Texto vacío"})
                continue

            logger.info(f"Iniciando síntesis: {texto[:50]}... ({idioma})")

            try:
                logger.info("Obteniendo latentes del speaker...")
                inicio_latentes = time.time()
                latente_gpt, embedding_speaker = modelo.get_conditioning_latents(
                    audio_path=[archivo_speaker]
                )
                tiempo_latentes = time.time() - inicio_latentes
                logger.info(f"Latentes obtenidos en {tiempo_latentes:.2f}s")
                logger.info(f"Shapes - latente_gpt: {latente_gpt.shape}, embedding: {embedding_speaker.shape}")

                logger.info("Iniciando inference_stream...")
                inicio_stream = time.time()
                chunks = modelo.inference_stream(
                    texto,
                    idioma,
                    latente_gpt,
                    embedding_speaker
                )

                cantidad_chunks = 0
                tiempo_chunk_anterior = inicio_stream

                for chunk in chunks:
                    tiempo_actual = time.time()
                    tiempo_desde_inicio = tiempo_actual - inicio_stream
                    tiempo_desde_chunk = tiempo_actual - tiempo_chunk_anterior

                    # Procesar chunk de audio
                    audio_bytes = procesar_chunk_audio(chunk)

                    logger.info(
                        f"Chunk {cantidad_chunks}: "
                        f"bytes={len(audio_bytes)}, "
                        f"tiempo_total={tiempo_desde_inicio:.2f}s, "
                        f"tiempo_chunk={tiempo_desde_chunk:.3f}s"
                    )

                    # Enviar chunk al cliente
                    await enviar_chunk(websocket, audio_bytes, cantidad_chunks)

                    cantidad_chunks += 1
                    tiempo_chunk_anterior = tiempo_actual

                tiempo_total = time.time() - inicio_stream
                logger.info(f"Stream completado: {cantidad_chunks} chunks en {tiempo_total:.2f}s")

                # Enviar JSON final con es_final: true
                await websocket.send_json({
                    "tipo": "completado",
                    "numero": cantidad_chunks,
                    "chunks_totales": cantidad_chunks,
                    "tiempo_total": round(tiempo_total, 2),
                    "es_final": True
                })

                logger.info(f"Síntesis completada exitosamente")

            except Exception as error:
                logger.error(f"Error en síntesis: {error}", exc_info=True)
                await websocket.send_json({"error": str(error)})

    except WebSocketDisconnect:
        logger.info(f"Cliente desconectado: {id_cliente}")
    except Exception as error:
        logger.error(f"Error: {error}")
        try:
            await websocket.send_json({"error": "Error en servidor"})
        except Exception:
            pass


# ============= PUNTO DE ENTRADA =============

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="info")