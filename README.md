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

a. Configurar el generador de señales biológicas en modo EMG, simulando
aproximadamente cinco contracciones musculares voluntarias.

b. Adquirir y almacenar la señal generada para su posterior análisis.

Para la adquisición de la señal se hizo uso de visual studio y spyder en donde se utilizó el siguiente codigo:

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


c. Segmentar la señal obtenida en las cinco contracciones simuladas.

d. Calcular para cada contracción:

 Frecuencia media

 Frecuencia mediana

e. Presentar los resultados de cada contracción en una tabla y representar
gráficamente la evolución de las frecuencias.

f. Analizar cómo varían estas frecuencias a lo largo de las contracciones
simuladas. 

PARTE B – Captura de la señal de paciente
a. Colocar los electrodos sobre el grupo muscular definido por el grupo (por
ejemplo, antebrazo o bíceps).

b. Registrar la señal EMG de un paciente o voluntario sano realizando
contracciones repetidas hasta la fatiga (o la falla).

c. Aplicar un filtro pasa banda (20–450 Hz) para eliminar ruido y artefactos.

d. Dividir la señal en el número de contracciones realizadas

e. Calcular para cada contracción:

 Frecuencia media

 Frecuencia mediana

f. Graficar los resultados obtenidos y analizar la tendencia de la frecuencia
media y mediana a medida que progresa la fatiga muscular.

g. Discutir la relación entre los cambios de frecuencia y la fisiología de la fatiga
muscular. 

PARTE C – Análisis espectral mediante FFT

a. Aplicar la Transformada Rápida de Fourier (FFT) a cada contracción de la
señal EMG real.

b. Graficar el espectro de amplitud (frecuencia vs. magnitud) para observar
cómo cambia el contenido de frecuencia.

c. Comparar los espectros de las primeras contracciones con los de las últimas.

d. Identificar la reducción del contenido de alta frecuencia asociada con la fatiga
muscular.

e. Calcular y discutir el desplazamiento del pico espectral y su relación con el
esfuerzo sostenido.

f. Redactar conclusiones sobre el uso del análisis espectral como herramienta
diagnóstica en electromiografía. 


