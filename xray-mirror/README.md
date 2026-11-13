# X-ray Mirror

Espejo de rayos X en el navegador: cámara + seguimiento de cara y manos (MediaPipe) + anatomía 3D real (BodyParts3D) fijada a tu cabeza con three.js. Todo se procesa en el dispositivo; no se sube video.

## Cómo abrirlo

La cámara en vivo solo funciona desde una dirección **https://** (o `localhost`). Abrir `index.html` con doble clic no da acceso a la cámara.

- **En la computadora:** dentro de esta carpeta ejecuta `python3 -m http.server 8000` y abre `http://localhost:8000`.
- **En iPhone y iPad:** sube la carpeta completa a un hosting estático con https (GitHub Pages, Netlify Drop, Cloudflare Pages) y abre esa dirección en Safari. Con "Compartir → Agregar a inicio" queda como app.

Sin cámara también funciona con una foto o un video (botón "Use a photo or video instead").

## Archivos

- `index.html` — interfaz y estilos
- `app.js` — toda la lógica (cámara, seguimiento, capas, lente, etiquetas, ajuste)
- `lib/` — three.js y MediaPipe Tasks Vision (incluye el .wasm)
- `assets/anatomy.bin` — mallas simplificadas: cráneo, mandíbula, cerebro, músculos, ojos y contorno de la cabeza
- `assets/*.task` — modelos de seguimiento de cara y manos

Parámetro útil: `?cpu=1` fuerza el seguimiento por CPU si el GPU da problemas en algún equipo.

## Créditos y licencias

- Anatomía: BodyParts3D, © The Database Center for Life Science, licensed under CC Attribution-Share Alike 2.1 Japan. Las mallas de `anatomy.bin` son una versión simplificada y se comparten bajo la misma licencia.
- MediaPipe Tasks Vision y sus modelos: Apache 2.0 (Google).
- three.js: MIT.

No es una herramienta médica: muestra dónde suelen estar las estructuras, no es un escaneo de la persona.
