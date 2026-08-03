# TrabajoGrado
Trabajo de grado

Instrucciones de instalación:

1) Instalar Unity Hub: https://docs.unity.com/en-us/hub/install-hub
2) Descargar la versión 6000.2.10f1 de Unity.
3) Crear nuevo proyecto 3D.
4) Copiar dentro de la carpeta Assets del proyecto todo el contenido de la carpeta Assets del repositorio
5) Ejecutar simulación.
   -Verificar que los componentes de NavAgent y NavMesh de unity estén funcionando.
   -En caso contrario, añadir los componentes de navegación y cocinar NavMesh en la escena
   -Escena a elegir se llama "Version1_P"

Controles de la cámara:
1) Click derecho en la ventana de Game
2) Cuando el crosshair sea visible:
   - Mirar -> Mouse
   - WASD -> Movimiento
   - Shift -> Aumentar velocidad
   - Disparar Raycast -> click izquierdo
   - Dirigir NPC a nueva posición -> click izquierdo a NPC y luego click izquierdo a nueva posición

Controles de la simulación:
Asignar personalidades, cantidad máxima de NPC y cantidad de NPC por personalidad en los campos dentro del inspector del GameManager
Enter - Inicializar agentes y comenzar simulación
E - Gol equipo rojo
Q - Gol equipo azul
V - Falta equipo rojo
B - Falta equipo azul
N - Tiro fallido equipo rojo
M - Tiro fallido equipo azul
F - Medio tiempo
ESPACIO - Fin del juego


Al finalizar la simulación, se genera un archivo .csv con los logs de la simulación en la raiz del proyecto, este archivo puede ser abierto en Excel o por medio del dashboard.py.
Para ejecutar dashboard.py, instalar librerías:
1) pip install streamlit pandas plotly scipy statsmodels numpy
2) abrir una terminal en la carpeta donde se encuentre dashboard.py y ejecutar: streamlit run dashboard.py
3) Una vez cargada la página, arrastrar el .csv con los logs de la simulación para visualizar gráficas.
