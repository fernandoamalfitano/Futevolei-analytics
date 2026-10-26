# Futevôlei Analytics: estadísticas automáticas de partidos con visión por computador

A partir del vídeo de un partido de futevôlei (*footvolley*), el sistema detecta el balón, los jugadores y la cancha, divide el partido en jugadas y calcula las estadísticas de cada equipo: **saques, aces, ataques y eficiencia de ataque**. Estas estadísticas se recogen normalmente a mano. Aquí lo que hacemos en automatizar el proceso.

Proyecto final del bootcamp de Big Data, Inteligencia Artificial y Machine Learning, desarrollado en equipo.

**Stack:** Python · YOLOv8 · YOLO11 (pose) · Roboflow · OpenCV · pandas · SciPy · PostgreSQL · Google Colab




https://github.com/user-attachments/assets/afe71e6e-7234-4150-8acd-133ac8d1e413

## Cómo funciona

```
Vídeo del partido
   │
   ├─ 1. Detección del balón y los jugadores (YOLOv8s) en cada fotograma
   ├─ 2. Trayectoria del balón: seguimiento propio, relleno de huecos y suavizado
   ├─ 3. Segmentación en jugadas: del saque al final del punto (sin repeticiones ni primeros planos)
   │
   ├─ Revisión manual: quitar jugadas falsas, añadir las que faltan e indicar los cambios de lado
   │
   ├─ 4. Puntos clave de la cancha (YOLO11-L pose) y homografía: de píxeles a metros
   ├─ 5. Toques, cruces de la red y resultado de cada golpe: punto, defendido o fuera
   │
   └─ 6. Estadísticas por equipo → tablas, vídeo con anotaciones y base de datos PostgreSQL
```

## Resultados y decisiones clave

- **Seguimiento del balón propio:** cubre el **85%** de los fotogramas, frente al 70% y el 65% de los trackers estándar (ByteTrack y BoT-SORT).
- **Homografía con solo 3 esquinas visibles:** cuando la cámara no ve la cancha entera, se estima el cuarto punto a partir de la última homografía válida y del giro de la cámara.
- **Sin fuga de datos:** el conjunto de entrenamiento, validación y prueba se separa por **vídeos completos**, porque los fotogramas consecutivos de un mismo vídeo son casi idénticos.
- **Aumentos de datos desactivados en el modelo de la cancha:** los volteos y recortes intercambiaban la identidad de las esquinas y enseñaban al modelo asociaciones incorrectas.
- **Estadísticas por equipo, no por jugador:** los trackers no mantenían la identidad de los jugadores tras cruces y oclusiones.

---

## Metodología

### Recolección y preparación de los datos

Se descargaron los videos de partidos de futevôlei y se editaron manualmente para conservar solo los segmentos con jugadas, eliminando pausas, repeticiones y tomas que no mostraban la cancha. Luego, mediante un script, se extrajeron los frames de cada video, y con un segundo script se filtraron los frames más útiles para el etiquetado, descartando los redundantes o de baja calidad.

El etiquetado se realizó en Roboflow.

Al construir el primer dataset cometimos un error en la partición: los conjuntos de entrenamiento, validación y prueba se generaron mezclando frames de todos los videos. Como los frames consecutivos de un mismo video son casi idénticos, el modelo se evaluaba sobre imágenes prácticamente iguales a las de entrenamiento (*data leakage*) y las métricas resultaban artificialmente altas. Para corregirlo, rehicimos la partición **por video**: todos los frames de un mismo video quedan en un único conjunto.

### Detección del balón y de los jugadores

Inicialmente etiquetamos solo el balón. Al incorporar a los jugadores vimos que no bastaba con etiquetarlos en imágenes nuevas: si en una imagen hay un jugador sin etiquetar, el modelo aprende que ese jugador es fondo. Por eso volvimos a las mismas imágenes y etiquetamos en cada una tanto el balón como los jugadores.

Los modelos se mejoraron con aumento de datos de dos tipos:

- **Offline en Roboflow:** genera copias modificadas de las imágenes al exportar el dataset.
- **Online de YOLO:** aplica transformaciones aleatorias durante el entrenamiento.

### Detección de los keypoints de la cancha

Etiquetamos 6 keypoints de la cancha (las 4 esquinas y los 2 extremos de la red) y entrenamos un modelo de estimación de keypoints (YOLO11-L pose).

El primer modelo dio malos resultados por los aumentos que YOLO aplica por defecto. Cada keypoint tiene una identidad fija (por ejemplo, "esquina superior izquierda"): un volteo horizontal convierte la esquina izquierda en derecha sin intercambiar los índices, y el mosaico o el escalado recortan o desplazan las esquinas fuera de la imagen. Así, el modelo aprendía asociaciones incorrectas. La solución fue desactivar los aumentos de YOLO:

```python
degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
flipud=0.0,
fliplr=0.0,    # evita invertir los índices izquierda/derecha
mosaic=0.0,    # evita recortar o deformar las esquinas
hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, erasing=0.0, mixup=0.0, copy_paste=0.0
```

### Seguimiento (tracking)

Probamos trackers (ByteTrack y BoT-SORT) para seguir el balón y los jugadores. Con los jugadores la detección era correcta, pero el tracker no conservaba la identidad: tras cruces u oclusiones, un jugador recibía un ID nuevo o intercambiaba su ID con otro. Como las estadísticas por jugador requieren una identidad estable, descartamos ese enfoque y las estadísticas se calculan por **equipo**.

Con el balón, los trackers perdían más detecciones que nuestro propio seguimiento (cobertura de la pelota de 85% a 70% y 65%), así que usamos un seguimiento propio de una sola pelota: en cada frame se elige la detección más cercana a la posición anterior (descartando saltos imposibles), se interpolan huecos cortos y se suaviza la trayectoria para obtener velocidades.

### Homografía de la cancha

Para llevar posiciones de la imagen a coordenadas reales de la cancha calculamos una homografía, que requiere como mínimo 4 puntos de correspondencia. Por el encuadre de la cámara, en muchos frames solo se detectan 3.

La solución:

1. Se recorren los frames hasta encontrar el primero con 4 o más keypoints; este sirve de referencia.
2. En los frames con solo 3 keypoints, se estima el 4º a partir de la última homografía válida, dentro de la mitad de la cancha que se está viendo, y se valida que el cuadrilátero resultante sea geométricamente plausible.
3. Los frames anteriores al primer frame de referencia se procesan en un segundo recorrido hacia atrás.

Como la cámara gira siguiendo la pelota, además se mide el paneo de la cámara entre frames. Con él se estabilizan las homografías y se ubica la red en cada frame, aunque no se vea.

### Segmentación del video en jugadas

Una función divide el video completo en jugadas. Antes se excluyen las repeticiones (detectando la animación amarilla que las abre y cierra) y los primeros planos. Una jugada comienza con el saque (pelota quieta que sale de los pies de un jugador, sube y avanza hacia la red) y termina con alguno de estos motivos:

- `ground`: el balón toca la arena.
- `caught`: un jugador atrapa el balón.
- `lost` / `lost_descending`: el balón deja de detectarse (en el segundo caso, mientras bajaba: probablemente cayó sin que se viera el contacto).
- `out_of_frame`: el balón sale del encuadre.
- `replay_cut` / `video_end`: la transmisión pasa a una repetición o el video termina.

### Cálculo de estadísticas

Una segunda función calcula, para cada jugada, de qué lado de la red está el balón en cada frame, los toques (cambios bruscos de velocidad que la gravedad no explica) y los cruces de la red. Con eso arma los golpes que cruzan la red (el saque y los ataques) y el resultado de cada uno: punto, defendido o fuera.

El ganador del punto se decide así:

- Si el balón cae y se ve con homografía: si cae **dentro**, gana el equipo contrario al lado donde cayó; si cae **fuera**, pierde el que lo tocó último.
- Si no se ve la caída: gana el equipo que realiza el saque siguiente, ya que en futevôlei saca quien ganó el punto.

Por defecto se asume que los equipos cambian de lado cada 6 puntos. Pero si la transmisión no muestra algún punto, la cuenta queda corrida. Por eso la función recibe una lista con las jugadas en que ocurre cada cambio (por ejemplo, 7, 13, …), que armamos revisando el video. También permite excluir jugadas mal segmentadas y agregar las que faltan.

Por lo tanto, el proceso no es completamente automático: tras generar el video segmentado, hay que revisarlo manualmente para indicar los cambios de lado y corregir las jugadas.

---

## Limitaciones y próximos pasos

- El proceso requiere una **revisión manual** de las jugadas antes de calcular las estadísticas.
- Las estadísticas son **por equipo**. Para obtenerlas por jugador haría falta un seguimiento que mantenga la identidad, por ejemplo con re-identificación por apariencia o por posición en la cancha.
- **Próximos pasos:** reducir la revisión manual, estadísticas por jugador y un panel de visualización a partir de la base de datos.

## Estructura del repositorio

```
├── futevolei_code/          # módulos del pipeline
│   ├── detection.py         # detección de balón y jugadores con YOLO
│   ├── ball_trajectory.py   # seguimiento y suavizado de la trayectoria del balón
│   ├── rallies.py           # segmentación del vídeo en jugadas
│   ├── camera.py            # movimiento de la cámara entre fotogramas
│   ├── homographies.py      # homografía de la cancha
│   ├── match_stats.py       # toques, golpes y estadísticas por equipo
│   ├── database.py          # guardado en PostgreSQL
│   └── pipeline.py          # orquestación de las etapas
├── models/                  # pesos entrenados (balón/jugadores y cancha)
├── notebooks/
│   └── futevolei_pipeline_final.ipynb   # pipeline completo, listo para Colab
├── training/                # entrenamiento de los dos modelos YOLO
├── sql/schema.sql           # esquema de la base de datos
└── requirements.txt
```

## Cómo ejecutarlo

1. Sube la carpeta `futevolei_code/` (o un .zip con ella) y los modelos de `models/` a Google Drive.
2. Abre `notebooks/futevolei_pipeline_final.ipynb` en Google Colab con GPU y ajusta `CODE_DIR` y las rutas del vídeo y los modelos.
3. **Paso 1:** detecta las jugadas y genera un vídeo de revisión con cada jugada numerada.
4. **Revisión:** anota las jugadas que hay que quitar o añadir y los cambios de lado.
5. **Paso 2:** calcula las estadísticas y genera el vídeo con anotaciones.
6. **Paso 3 (opcional):** guarda los resultados en PostgreSQL (`sql/schema.sql`).

## Equipo

- **Fernando Amalfitano**: [LinkedIn](https://www.linkedin.com/in/fernando-amalfitano-051751388/)
- **Ignacio Emanuelli**
- **Cristian David Noreña Saldarriaga**
