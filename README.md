# Senales-electromiograficas-EMG

OBJETIVOS

 Aplicar el filtrado de señales continuas para procesar una señal electromiográfica
(EMG).

 Detectar la aparición de fatiga muscular mediante el análisis espectral de
contracciones musculares individuales.

 Comparar el comportamiento de una señal emulada y una señal real en términos
de frecuencia media y mediana.

 Emplear herramientas computacionales para el procesamiento, segmentación y
análisis de señales biomédicas.

PARTE A – Captura de la señal emulada

a) Configurar el generador de señales biológicas en modo EMG, simulando
aproximadamente cinco contracciones musculares voluntarias.

b) Adquirir y almacenar la señal generada para su posterior análisis.

Para la adquisición de la señal se hizo uso de visual studio y spyder para la creación del código de captura de la señal EMG del generador de señales biológicas, en donde se diseña un filtro FIR hamming, ya que este es un tipo de filtro muy común para señales biológicas por su buen compromiso entre selectividad y atenuación en banda de rechazo. Para la creación del filtro se siguieron los pasos indicados en teoría hasta llegar al calculo del orden debido a que se decidió hacer uso de la app de matlab filter designer para exportar los coeficientes y de esta manera concluir el diseño del filtro.

```
import numpy as np  
import nidaqmx
from nidaqmx.constants import AcquisitionType, READ_ALL_AVAILABLE
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

rate = 2000.0  # Hz
duration = 5.5  # segundos

samps_per_chan = int(rate * duration)
corte_alto = 500 #hz
corte_bajo = 20 #hz
corte_altorad = corte_alto * 2 * np.pi
corte_bajorad = corte_bajo * 2 * np.pi
w2=corte_bajorad/rate
w1=corte_altorad/rate
N = (2*np.pi*4)/w2-w1
print(w1)
print(w2)
print(N)
b = np.loadtxt('C:/Users/USUARIO/Downloads/coeflab4.txt', delimiter=',')
a = np.array([1])  # porque es un filtro FIR (Hamming)



with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan("Dev5/ai0")
    task.timing.cfg_samp_clk_timing(rate, sample_mode=AcquisitionType.FINITE, samps_per_chan=samps_per_chan)
    data = task.read(READ_ALL_AVAILABLE)
    #offset = 10
    data = np.array(data)
    data_filtrada = filtfilt(b, a, data)
    #dataoffset = data+offset
    plt.figure(figsize=(15, 3))
    plt.plot(data_filtrada)
    plt.ylabel('Amplitud')
    #plt.ylim(-1, 1)
    plt.title('señal emg')  
    plt.show()


np.savetxt(r"C:\Users\USUARIO\OneDrive\Desktop\datosEMGgeneradordeseñales.csv", data, delimiter=",")
```

Posterior a su captura, se guardo en drive en una carpeta la señal capturada para luego graficarla en google colab y poder observar las 5 contracciones voluntarias simuladas.

```
from google.colab import drive
drive.mount('/content/drive')
from google.colab import drive
import pandas as pd
import matplotlib.pyplot as plt

drive.mount('/content/drive') 

try:
  graficacaptura = pd.read_excel("/content/drive/Shareddrives/Labs procesamiento de señales/lab 4/datosEMGgeneradordeseñales.xlsx")


  if not graficacaptura.empty:
    for column in graficacaptura.columns:

        plt.figure(figsize=(15, 4))
        plt.xlabel("(2000muestras/seg)")
        plt.ylabel("Amplitud (V)")
        plt.title("Señal EMG Generada")
        plt.plot(graficacaptura[column], label=f"Señal {column}")
        plt.legend()
        plt.show()
  else:
    print("The Excel file is empty or could not be read.")

except FileNotFoundError:
  print("Error: The file was not found. Please check the file path.")
except Exception as e:
  print(f"An error occurred: {e}")
```


<img width="1232" height="393" alt="image" src="https://github.com/user-attachments/assets/2d883117-ca7d-4ee9-ac46-68d44c7a6f5a" />


c) Segmentar la señal obtenida en las cinco contracciones simuladas.

Para segementar la señal en las cinco contracciones, se hizo uso de algo similar al funcionamiento del metodo de banderas o flag method, el cual se basa en marcar ciertos eventos o condiciones dentro de una señal o conjunto de datos mediante banderas lógicas, que indican si algo cumple un criterio o no. En este caso como se quiere segmentar las 5 contracciones lo que se hizo fue encender una bandera o marca para todos los valores positivos como se muestra a continuación:

```
positive_data = data.copy()
positive_data[positive_data < 0] = 0  # Considerar solo valores positivos

threshold = 0.95 * np.max(positive_data)

```

d) Calcular para cada contracción:

 Frecuencia media

 Frecuencia mediana

e) Presentar los resultados de cada contracción en una tabla y representar
gráficamente la evolución de las frecuencias.

A continuación se muestra el código completo en donde se evidencia la segmentación de las 5 contracciones simuladas con sus respectivas frecuencias medias y medianas:

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

# Cargar datos
data = np.loadtxt('/content/drive/Shareddrives/Labs procesamiento de señales/lab 4/datosEMGgeneradordeseñales.csv')

# **Paso 1: Identificar picos POSITIVOS solamente**
# Umbral relativo al máximo positivo (ajustable)
positive_data = data.copy()
positive_data[positive_data < 0] = 0  # Considerar solo valores positivos

threshold = 0.95 * np.max(positive_data)

# Encontrar picos solo en la parte positiva
peaks, properties = find_peaks(positive_data, height=threshold, distance=100)

print(f"Número de picos positivos encontrados: {len(peaks)}")
print(f"Posiciones de los picos: {peaks}")
print(f"Alturas de los picos: {properties['peak_heights']}")

# **Paso 2: Segmentar cada contracción basada en picos positivos**
window_size = 200  # Ajustar según la duración típica de una contracción
segments = []

for peak in peaks:
    start = max(0, peak - window_size)
    end = min(len(data), peak + window_size)
    segments.append(data[start:end])

print(f"Segmentos extraídos: {len(segments)}")

# **Paso 3: Calcular frecuencia media y mediana por segmento**
freq_means = []
freq_medians = []
sampling_rate = 1000  # Hz (ajustar según tu frecuencia de muestreo real)

for i, segment in enumerate(segments):
    # FFT
    N = len(segment)
    if N == 0:
        continue

    T = 1 / sampling_rate
    yf = fft(segment)
    xf = fftfreq(N, T)[:N//2]

    # Densidad espectral de potencia (PSD)
    psd = np.abs(yf[:N//2])**2

    # Excluir frecuencia DC (0 Hz) para análisis
    mask = xf > 0
    xf_filtered = xf[mask]
    psd_filtered = psd[mask]

    if len(xf_filtered) == 0 or np.sum(psd_filtered) == 0:
        freq_means.append(0)
        freq_medians.append(0)
        continue

    # Frecuencia media (ponderada por PSD)
    mean_freq = np.sum(xf_filtered * psd_filtered) / np.sum(psd_filtered)

    # Frecuencia mediana (frecuencia donde la PSD acumulada alcanza la mitad)
    cumsum_psd = np.cumsum(psd_filtered)
    median_idx = np.where(cumsum_psd >= cumsum_psd[-1] / 2)[0]
    if len(median_idx) > 0:
        median_freq = xf_filtered[median_idx[0]]
    else:
        median_freq = 0

    freq_means.append(mean_freq)
    freq_medians.append(median_freq)

# **Visualización de la señal segmentada (picos positivos)**
plt.figure(figsize=(14, 8))

# Señal completa
plt.subplot(2, 1, 1)
plt.plot(data, label='Señal EMG completa', alpha=0.7)
plt.axhline(y=threshold, color='r', linestyle='--', label='Umbral de picos positivos')
plt.plot(peaks, data[peaks], 'ro', markersize=8, label='Picos positivos detectados')
plt.xlabel('Muestras')
plt.ylabel('Amplitud (V)')
plt.legend()
plt.title('Detección de picos positivos en señal EMG')
plt.grid(True)

# Segmentos individuales
plt.subplot(2, 1, 2)
for i, segment in enumerate(segments):
    time_axis = np.arange(len(segment)) / sampling_rate * 1000  # ms
    plt.plot(time_axis, segment, label=f'Contracción {i+1}')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Amplitud (V)')
plt.legend()
plt.title('Segmentos de contracción extraídos')
plt.grid(True)

plt.tight_layout()
plt.show()

#**Resultados**
print("\n" + "="*60)
print("RESULTADOS DE ANÁLISIS DE FRECUENCIA POR CONTRACCIÓN")
print("="*60)

for i, (mean_f, median_f) in enumerate(zip(freq_means, freq_medians)):
    print(f"Contracción {i+1}:")
    print(f"  • Frecuencia media = {mean_f:.2f} Hz")
    print(f"  • Frecuencia mediana = {median_f:.2f} Hz")
    print(f"  • Duración = {len(segments[i])/sampling_rate*1000:.1f} ms")
    print()

# Estadísticas generales
print("ESTADÍSTICAS GENERALES:")
print(f"Frecuencia media promedio: {np.mean(freq_means):.2f} Hz")
print(f"Frecuencia mediana promedio: {np.mean(freq_medians):.2f} Hz")
print(f"Desviación estándar frecuencia media: {np.std(freq_means):.2f} Hz")

# **Evolución de las frecuencias por contracción**
plt.figure(figsize=(10, 5))

# Gráfico de evolución de la frecuencia media
plt.plot(range(1, len(freq_means) + 1), freq_means, 'o-', label='Frecuencia media (Hz)')
plt.plot(range(1, len(freq_medians) + 1), freq_medians, 's--', label='Frecuencia mediana (Hz)')

plt.title('Evolución de la frecuencia media y mediana por contracción')
plt.xlabel('Número de contracción')
plt.ylabel('Frecuencia (Hz)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

```


<img width="1389" height="790" alt="image" src="https://github.com/user-attachments/assets/78c91e91-9ade-408a-af58-fbe7a87a5dc8" />


```
============================================================
RESULTADOS DE ANÁLISIS DE FRECUENCIA POR CONTRACCIÓN
============================================================
Contracción 1:
  • Frecuencia media = 7.75 Hz
  • Frecuencia mediana = 2.50 Hz
  • Duración = 400.0 ms

Contracción 2:
  • Frecuencia media = 7.78 Hz
  • Frecuencia mediana = 2.50 Hz
  • Duración = 400.0 ms

Contracción 3:
  • Frecuencia media = 7.84 Hz
  • Frecuencia mediana = 2.50 Hz
  • Duración = 400.0 ms

Contracción 4:
  • Frecuencia media = 7.85 Hz
  • Frecuencia mediana = 2.50 Hz
  • Duración = 400.0 ms

Contracción 5:
  • Frecuencia media = 7.87 Hz
  • Frecuencia mediana = 2.50 Hz
  • Duración = 400.0 ms

ESTADÍSTICAS GENERALES:
Frecuencia media promedio: 7.82 Hz
Frecuencia mediana promedio: 2.50 Hz
Desviación estándar frecuencia media: 0.05 Hz
```


<img width="989" height="490" alt="image" src="https://github.com/user-attachments/assets/25e8cfce-482f-4d44-a509-bbaedfbc807a" />



f) Analizar cómo varían estas frecuencias a lo largo de las contracciones
simuladas. 


Como se observa en la grafica de segmentos de contracción extraidos, las contracciones están sobrepuestas, y tiene todo el sentido debido a que la señal EMG que se está analizando, es una señal simulada y generada por un generador de señales lo cual indica que al segmentar la señal en las cinco contracciones y compararlas son parecidas.


PARTE B – Captura de la señal de paciente

a) Colocar los electrodos sobre el grupo muscular definido por el grupo (por
ejemplo, antebrazo o bíceps).

El grupo muscular seleccionado fue el biceps, a continuación se adjuntan imágenes del posicionamiento de los electrodos para la captura de la señal EMG.


![Imagen de WhatsApp 2025-11-05 a las 21 40 05_a93ce64a](https://github.com/user-attachments/assets/eaa074d9-1387-425d-b9e2-e5cae47e04a1)


![Imagen de WhatsApp 2025-11-05 a las 21 40 05_cb674ad4](https://github.com/user-attachments/assets/036171b4-466e-4b51-90d7-f00b965cfa01)



b) Registrar la señal EMG de un paciente o voluntario sano realizando
contracciones repetidas hasta la fatiga (o la falla).
c) Aplicar un filtro pasa banda (20–450 Hz) para eliminar ruido y artefactos.
Para este caso, se utilizó el mismo filtro que ya se habia diseñado en la parte A para la captura de la señal EMG del generador de señales biológicas,
```
import numpy as np  
import nidaqmx
from nidaqmx.constants import AcquisitionType, READ_ALL_AVAILABLE
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

rate = 2000.0  # Hz
duration = 6  # segundos

samps_per_chan = int(rate * duration)
corte_alto = 500 #hz
corte_bajo = 20 #hz
corte_altorad = corte_alto * 2 * np.pi
corte_bajorad = corte_bajo * 2 * np.pi
w2=corte_bajorad/rate
w1=corte_altorad/rate
N = (2*np.pi*4)/w2-w1
print(w1)
print(w2)
print(N)
b = np.loadtxt('C:/Users/USUARIO/Downloads/coeflab4.txt', delimiter=',')
a = np.array([1])  # porque es un filtro FIR (Hamming)



with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan("Dev1/ai0")
    task.timing.cfg_samp_clk_timing(rate, sample_mode=AcquisitionType.FINITE, samps_per_chan=samps_per_chan)
    data = task.read(READ_ALL_AVAILABLE)
    #offset = 10
    data = np.array(data)
    data_filtrada = filtfilt(b, a, data)
    #dataoffset = data+offset
    plt.figure(figsize=(15, 3))
    plt.plot(data_filtrada)
    plt.ylabel('Amplitud')
    #plt.ylim(-1, 1)
    plt.title('señal emg')  
    plt.show()


np.savetxt(r"C:\Users\USUARIO\OneDrive\Desktop\DATOSPARTEBEMGintentofinal.csv", data_filtrada, delimiter=",")
```

d) Dividir la señal en el número de contracciones realizadas
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

filepath = '/content/drive/Shareddrives/Labs procesamiento de señales/lab 4/DATOSPARTEBEMGintentofinal.csv'
sampling_rate = 2000  # Hz
threshold_factor = 0.50   # comparacion con el máximo positivo que consideramos "umbral"
window_ms_group = 900  # ventana fija para agrupar picos en ms (cada ventana es 1 contracción maxima)
segment_ms = 800         # ancho (antes+después) para extraer segmentos alrededor del pico 

# Cargar datos
data = np.loadtxt(filepath)

# Considerar solo parte positiva para detección de picos
positive_data = data.copy()
positive_data[positive_data < 0] = 0

# Umbral relativo al máximo positivo (ajustable)
threshold = threshold_factor * np.max(positive_data)

# Encontrar todos los picos con un umbral relativamente bajo (para no perder picos pequeños)
all_peaks, all_props = find_peaks(positive_data, height=threshold/2, distance=10)  # distance pequeña solo para limpieza
# (usamos un umbral algo bajo aquí porque luego dentro de la ventana elegimos el pico más alto)

# Agrupar por ventanas de tiempo fijas y seleccionar el pico más alto por ventana
samples_per_group = max(1, int(window_ms_group * sampling_rate / 1000))
selected_peaks = []

start = 0
n = len(data)
while start < n:
    end = min(start + samples_per_group, n)
    # picos dentro de esta ventana (indices relativos a la señal completa)
    mask = (all_peaks >= start) & (all_peaks < end)
    peaks_in_win = all_peaks[mask]
    if peaks_in_win.size > 0:
        heights = positive_data[peaks_in_win]
        # escoger el pico con mayor altura dentro de la ventana
        chosen_idx = np.argmax(heights)
        selected_peaks.append(peaks_in_win[chosen_idx])
    start += samples_per_group

selected_peaks = np.array(selected_peaks, dtype=int)

print(f"Total picos detectados (preliminar): {len(all_peaks)}")
print(f"Picos seleccionados (1 por ventana de {window_ms_group} ms): {len(selected_peaks)}")
print("Posiciones de picos seleccionados:", selected_peaks)

# Extraer segmentos alrededor de cada pico seleccionado
half_segment_samples = max(1, int(segment_ms * sampling_rate / 1000))
segments = []
for peak in selected_peaks:
    s = max(0, peak - half_segment_samples)
    e = min(n, peak + half_segment_samples)
    segments.append(data[s:e])

print(f"Segmentos extraídos: {len(segments)}")
```
<img width="1390" height="790" alt="image" src="https://github.com/user-attachments/assets/794a8762-c3d0-494b-8fad-bfa92cb72bb8" />

e.) Calcular para cada contracción:

 Frecuencia media

 Frecuencia mediana

```
freq_means = []
freq_medians = []

for i, segment in enumerate(segments):
    N = len(segment)
    if N == 0:
        freq_means.append(0)
        freq_medians.append(0)
        continue

    T = 1 / sampling_rate
    yf = fft(segment)
    xf = fftfreq(N, T)[:N//2]
    psd = np.abs(yf[:N//2])**2

    mask = xf > 0
    xf_filtered = xf[mask]
    psd_filtered = psd[mask]

    if len(xf_filtered) == 0 or np.sum(psd_filtered) == 0:
        freq_means.append(0)
        freq_medians.append(0)
        continue

    mean_freq = np.sum(xf_filtered * psd_filtered) / np.sum(psd_filtered)
    cumsum_psd = np.cumsum(psd_filtered)
    median_idx = np.where(cumsum_psd >= cumsum_psd[-1] / 2)[0]
    median_freq = xf_filtered[median_idx[0]] if len(median_idx) > 0 else 0

    freq_means.append(mean_freq)
    freq_medians.append(median_freq)
```
<img width="488" height="657" alt="image" src="https://github.com/user-attachments/assets/b3e10909-a3d7-46fe-9346-d2c3c8a4e8c5" />



f) Graficar los resultados obtenidos y analizar la tendencia de la frecuencia
media y mediana a medida que progresa la fatiga muscular.
```
plt.figure(figsize=(14, 8))

# Señal completa + umbral + todos los picos preliminares (pequeños) + picos seleccionados (grandes)
plt.subplot(2, 1, 1)
plt.plot(data, label='Señal EMG completa', alpha=0.7)
plt.axhline(y=threshold, color='r', linestyle='--', label='Umbral (threshold_factor * max positivo)')
# marcar picos preliminares con puntos pequeños y semitransparentes
plt.plot(all_peaks, data[all_peaks], 'k.', markersize=4, alpha=0.3, label='Picos preliminares')
# marcar picos seleccionados (1 por ventana) con rojos grandes
plt.plot(selected_peaks, data[selected_peaks], 'ro', markersize=8, label='Picos seleccionados (1/ventana)')
# Dibujar división de ventanas para que veas cómo se agrupan
for x in range(0, n, samples_per_group):
    plt.axvline(x=x, color='gray', alpha=0.15)

plt.xlabel('Muestras')
plt.ylabel('Amplitud (V)')
plt.legend()
plt.title('Detección de picos: 1 pico máximo por ventana fija')
plt.grid(True)

# Segmentos individuales (tiempo en ms)
plt.subplot(2, 1, 2)
for i, segment in enumerate(segments):
    time_axis = np.arange(len(segment)) / sampling_rate * 1000  # ms
    plt.plot(time_axis, segment, label=f'Contracción {i+1}')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Amplitud (V)')
plt.legend()
plt.title('Segmentos de contracción extraídos (alrededor del pico seleccionado)')
plt.grid(True)

plt.tight_layout()
plt.show()

# ----- Resultados -----
print("\n" + "="*60)
print("RESULTADOS DE ANÁLISIS DE FRECUENCIA POR CONTRACCIÓN")
print("="*60)
for i, (mean_f, median_f) in enumerate(zip(freq_means, freq_medians)):
    dur_ms = len(segments[i]) / sampling_rate * 1000 if len(segments[i]) > 0 else 0
    print(f"Contracción {i+1}:")
    print(f"  • Frecuencia media = {mean_f:.2f} Hz")
    print(f"  • Frecuencia mediana = {median_f:.2f} Hz")
    print(f"  • Duración = {dur_ms:.1f} ms")
    print()

print("ESTADÍSTICAS GENERALES:")
print(f"Frecuencia media promedio: {np.mean(freq_means):.2f} Hz")
print(f"Frecuencia mediana promedio: {np.mean(freq_medians):.2f} Hz")
print(f"Desviación estándar frecuencia media: {np.std(freq_means):.2f} Hz")


fig = plt.figure(figsize=(12,8))
ax3 = fig.add_subplot(111, projection='3d')

# Parámetros de separación y escala
n_segments = len(segments)
separation = 1.0                    # separación entre "capas" en el eje Y
amplitude_scale = 1.0               # escala global de amplitud (ajustar si es necesario)
cm_map = cm.get_cmap('viridis', max(2, n_segments))

for i, segment in enumerate(segments):
    N = len(segment)
    if N == 0:
        continue
    # eje X: tiempo relativo en ms
    x = np.arange(N) / sampling_rate * 1000.0
    # eje Y: colocar cada contracción en su "capa" usando su frecuencia media si disponible,
    # sino usar el índice para separarlas de forma uniforme
    if i < len(freq_means) and freq_means[i] > 0:
        y_level = freq_means[i]  # posicionar por frecuencia media (Hz)
    else:
        y_level = i * separation  # fallback por índice

    # crear arrays 3D: y constante para toda la curva, z = amplitud
    y = np.full_like(x, fill_value=y_level, dtype=float)
    z = segment * amplitude_scale

    # Trazo principal (línea) y sombreado ligero con transparencia
    ax3.plot(x, y, z, linewidth=2.2, color=cm_map(i), label=f'Contracción {i+1}' if i<10 else None)
    ax3.plot(x, y, z, linewidth=6, alpha=0.06, color='k')  # halo tenue para mejor contraste

    # marcar el pico seleccionado dentro del segmento (posición 0 corresponde al inicio del segmento)
    # buscar el índice del valor máximo dentro del segmento y representarlo
    peak_idx = np.argmax(np.abs(z))
    ax3.scatter(x[peak_idx], y[peak_idx], z[peak_idx], s=30, color=cm_map(i), edgecolor='k')

# Ajustes estéticos
ax3.set_xlabel('Tiempo (ms)')
ax3.set_ylabel('Frecuencia media (Hz) / Nivel')
ax3.set_zlabel('Amplitud (V)')
ax3.view_init(elev=30, azim=45)   # vista isométrica aproximada (ajustar elev/azim a tu gusto)
ax3.grid(True)
plt.tight_layout()
plt.show()
```

<img width="983" height="790" alt="image" src="https://github.com/user-attachments/assets/b5b91d2b-3a1f-4217-a288-5467247b8f39" />



g) Discutir la relación entre los cambios de frecuencia y la fisiología de la fatiga
muscular. 
no se pudo identificar informacion importante para identificar fatiga muscular durante el examen, debido a el corto tiempo de el examen, la ausencia de peso durante, por eso es que no se ven desplazamientos significativos de los espectros de cada contraccion, sin embargo podemos asumir que el musculo ya estaba con fatiga por las multiples contracciones y ejercicios que se realizaron antes de la toma de los datos, pues con cada ajuste de parametros y captura se seguia realizando contracciones continuamente hasta que llego el punto donde obtuvimos señales de la forma en que se graficaron donde se ve mayor actividad muscular
A continuación se presenta el siguiente codigo que se utilizó para dar solución a todos los items que exigía la parte C de esta práctica de laboratorio.


PARTE C – Análisis espectral mediante FFT

A continuación se presenta el siguiente codigo que se utilizó para dar solución a todos los items que exigía la parte C de esta práctica de laboratorio.

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

# ==========================
# CARGAR DATOS
# ==========================
data = np.loadtxt('/content/drive/Shareddrives/Labs procesamiento de señales/lab 4/DATOSPARTEBEMGintentofinal.csv')

# ==========================
# DETECTAR PICOS POSITIVOS
# ==========================
positive_data = data.copy()
positive_data[positive_data < 0] = 0  # solo valores positivos

threshold = 0.63 * np.max(positive_data)
peaks, properties = find_peaks(positive_data, height=threshold, distance=100)

print(f"\nNúmero de picos positivos encontrados: {len(peaks)}")

# ==========================
# SEGMENTAR CADA CONTRACCIÓN
# ==========================
window_size = 200  # ajustar según la duración típica
segments = []
for peak in peaks:
    start = max(0, peak - window_size)
    end = min(len(data), peak + window_size)
    segments.append(data[start:end])

sampling_rate = 2000  # Hz

# ==========================
# FFT Y ANÁLISIS DE CADA CONTRACCIÓN
# ==========================
freq_means = []
freq_medians = []

num_contracciones = len(segments)

# Crear una figura con subplots bien distribuidos
fig, axs = plt.subplots(num_contracciones, 1, figsize=(10, 3*num_contracciones))
if num_contracciones == 1:
    axs = [axs]  # por si solo hay una contracción

for i, segment in enumerate(segments):
    N = len(segment)
    if N == 0:
        continue

    T = 1 / sampling_rate
    yf = fft(segment)
    xf = fftfreq(N, T)[:N//2]
    psd = np.abs(yf[:N//2])**2

    # Filtrar DC
    mask = xf > 0
    xf_filtered = xf[mask]
    psd_filtered = psd[mask]

    # Frecuencia media y mediana
    mean_freq = np.sum(xf_filtered * psd_filtered) / np.sum(psd_filtered)
    cumsum_psd = np.cumsum(psd_filtered)
    median_idx = np.where(cumsum_psd >= cumsum_psd[-1] / 2)[0]
    median_freq = xf_filtered[median_idx[0]] if len(median_idx) > 0 else 0

    freq_means.append(mean_freq)
    freq_medians.append(median_freq)

    # Graficar cada espectro
    axs[i].plot(xf_filtered, psd_filtered, label=f'Contracción {i+1}', color='blue')
    axs[i].axvline(mean_freq, color='r', linestyle='--', label=f'Media: {mean_freq:.2f} Hz')
    axs[i].axvline(median_freq, color='g', linestyle='--', label=f'Mediana: {median_freq:.2f} Hz')
    axs[i].set_title(f'Espectro de Frecuencia - Contracción {i+1}')
    axs[i].set_xlabel('Frecuencia (Hz)')
    axs[i].set_ylabel('Potencia (u.a.)')
    axs[i].legend(loc='upper right')
    axs[i].grid(True)

plt.tight_layout()
plt.show()

# ==========================
# RESULTADOS NUMÉRICOS
# ==========================
print("\n" + "="*60)
print("RESULTADOS DE ANÁLISIS DE FRECUENCIA POR CONTRACCIÓN")
print("="*60)
for i, (mean_f, median_f) in enumerate(zip(freq_means, freq_medians)):
    print(f"Contracción {i+1}:")
    print(f"  • Frecuencia media = {mean_f:.2f} Hz")
    print(f"  • Frecuencia mediana = {median_f:.2f} Hz")
    print(f"  • Duración = {len(segments[i])/sampling_rate*1000:.1f} ms")
    print()
    # ===========================================
# GRAFICAR ESPECTRO DE AMPLITUD Y COMPARAR INICIALES VS FINALES
# ===========================================

# Calcular espectro de amplitud (no potencia) para comparación visual
plt.figure(figsize=(12, 6))

# Seleccionar primeras y últimas contracciones (por ejemplo 1-2 vs 6-7)
primeras = segments[:2]
ultimas = segments[-2:]

# --- Graficar espectros de las primeras contracciones ---
for i, segment in enumerate(primeras):
    N = len(segment)
    T = 1 / sampling_rate
    yf = fft(segment)
    xf = fftfreq(N, T)[:N//2]
    amplitude_spectrum = np.abs(yf[:N//2]) / N  # espectro de amplitud normalizado
    plt.plot(xf, amplitude_spectrum, label=f'Contracción inicial {i+1}')

# --- Graficar espectros de las últimas contracciones ---
for i, segment in enumerate(ultimas):
    N = len(segment)
    T = 1 / sampling_rate
    yf = fft(segment)
    xf = fftfreq(N, T)[:N//2]
    amplitude_spectrum = np.abs(yf[:N//2]) / N
    plt.plot(xf, amplitude_spectrum, linestyle='--', label=f'Contracción final {len(segments)-1+i}')

plt.title('Comparación de Espectros de Amplitud: Primeras vs Últimas Contracciones')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud (u.a.)')
plt.legend()
plt.grid(True)
plt.xlim(0, 500)  # limitar eje de frecuencia para ver zona útil
plt.show()
# ===========================================
# ANÁLISIS DE FATIGA MUSCULAR
# ===========================================

high_freq_band = (80, 150)  # rango típico de altas frecuencias en EMG
peak_freqs = []
high_freq_energy = []

for i, segment in enumerate(segments):
    N = len(segment)
    T = 1 / sampling_rate
    yf = fft(segment)
    xf = fftfreq(N, T)[:N//2]
    amplitude_spectrum = np.abs(yf[:N//2]) / N

    # --- Calcular frecuencia del pico espectral (máxima amplitud) ---
    peak_idx = np.argmax(amplitude_spectrum)
    peak_freq = xf[peak_idx]
    peak_freqs.append(peak_freq)

    # --- Calcular energía en banda alta (80–150 Hz) ---
    mask_high = (xf >= high_freq_band[0]) & (xf <= high_freq_band[1])
    high_energy = np.sum(amplitude_spectrum[mask_high])
    high_freq_energy.append(high_energy)

# ===========================================
# MOSTRAR RESULTADOS
# ===========================================
print("\n" + "="*65)
print("ANÁLISIS DE FATIGA MUSCULAR")
print("="*65)
for i, (pf, hf) in enumerate(zip(peak_freqs, high_freq_energy)):
    print(f"Contracción {i+1}:")
    print(f"  • Frecuencia pico = {pf:.2f} Hz")
    print(f"  • Energía en banda alta (80–150 Hz) = {hf:.4f}")
    print()

# ===========================================
# VISUALIZAR TENDENCIA DE FATIGA
# ===========================================
plt.figure(figsize=(10,5))
plt.subplot(2,1,1)
plt.plot(range(1, len(segments)+1), high_freq_energy, 'o-', color='tab:red')
plt.title('Reducción del contenido de alta frecuencia (80–150 Hz)')
plt.xlabel('Número de Contracción')
plt.ylabel('Energía relativa (u.a.)')
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(range(1, len(segments)+1), peak_freqs, 'o-', color='tab:blue')
plt.title('Desplazamiento del pico espectral')
plt.xlabel('Número de Contracción')
plt.ylabel('Frecuencia pico (Hz)')
plt.grid(True)

plt.tight_layout()
plt.show()



```

a) Aplicar la Transformada Rápida de Fourier (FFT) a cada contracción de la
señal EMG real.

La FFT se utiliza para transformar la señal EMG del dominio del tiempo al dominio de la frecuencia para estudiar la distribución de la energía de la señal en distintas frecuencias durante cada contracción muscular. 

<img width="989" height="2090" alt="image" src="https://github.com/user-attachments/assets/b99e5783-992b-4286-9c78-afe0e03f3593" />


b) Graficar el espectro de amplitud (frecuencia vs. magnitud) para observar
cómo cambia el contenido de frecuencia.

c) Comparar los espectros de las primeras contracciones con los de las últimas.


<img width="1023" height="550" alt="image" src="https://github.com/user-attachments/assets/2ed55359-f7ec-4675-babf-ea30a979ac59" />


d) Identificar la reducción del contenido de alta frecuencia asociada con la fatiga
muscular.

```
=================================================================
ANÁLISIS DE FATIGA MUSCULAR
=================================================================
Contracción 1:
  • Frecuencia pico = 40.00 Hz
  • Energía en banda alta (80–150 Hz) = 0.0952

Contracción 2:
  • Frecuencia pico = 40.00 Hz
  • Energía en banda alta (80–150 Hz) = 0.0963

Contracción 3:
  • Frecuencia pico = 40.00 Hz
  • Energía en banda alta (80–150 Hz) = 0.0846

Contracción 4:
  • Frecuencia pico = 35.00 Hz
  • Energía en banda alta (80–150 Hz) = 0.0854

Contracción 5:
  • Frecuencia pico = 30.00 Hz
  • Energía en banda alta (80–150 Hz) = 0.0829

Contracción 6:
  • Frecuencia pico = 35.00 Hz
  • Energía en banda alta (80–150 Hz) = 0.0945

Contracción 7:
  • Frecuencia pico = 30.00 Hz
  • Energía en banda alta (80–150 Hz) = 0.0764
```

e) Calcular y discutir el desplazamiento del pico espectral y su relación con el
esfuerzo sostenido.

<img width="819" height="404" alt="image" src="https://github.com/user-attachments/assets/50e57081-a194-4607-94bb-2c80d3fccec4" />


f) Redactar conclusiones sobre el uso del análisis espectral como herramienta
diagnóstica en electromiografía. 


