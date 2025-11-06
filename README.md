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

c) Aplicar un filtro pasa banda (20–450 Hz) para eliminar ruido y artefactos.


Para este caso, se utilizó el mismo filtro que ya se habia diseñado en la parte A para la captura de la señal EMG del generador de señales biológicas,


d) Dividir la señal en el número de contracciones realizadas

e.) Calcular para cada contracción:

 Frecuencia media

 Frecuencia mediana

f) Graficar los resultados obtenidos y analizar la tendencia de la frecuencia
media y mediana a medida que progresa la fatiga muscular.

g) Discutir la relación entre los cambios de frecuencia y la fisiología de la fatiga
muscular. 

PARTE C – Análisis espectral mediante FFT

a) Aplicar la Transformada Rápida de Fourier (FFT) a cada contracción de la
señal EMG real.

b) Graficar el espectro de amplitud (frecuencia vs. magnitud) para observar
cómo cambia el contenido de frecuencia.

c) Comparar los espectros de las primeras contracciones con los de las últimas.

d) Identificar la reducción del contenido de alta frecuencia asociada con la fatiga
muscular.

e) Calcular y discutir el desplazamiento del pico espectral y su relación con el
esfuerzo sostenido.

f) Redactar conclusiones sobre el uso del análisis espectral como herramienta
diagnóstica en electromiografía. 


