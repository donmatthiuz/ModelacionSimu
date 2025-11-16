# Simulación de la Dispersión Atmosférica del Desastre de Chernobyl

## Explicación del Fenómeno a Simular

El accidente de Chernobyl (26 de abril de 1986) liberó grandes cantidades de radionúclidos (p. ej. Cs-137, Sr-90, I-131). La simulación reproduce el transporte tridimensional de estos trazadores desde la fuente, considerando advección por el viento, difusión turbulenta, mezcla vertical/horizontal, deposición seca y húmeda, y decaimiento radiactivo. El objetivo es reproducir patrones de concentración y deposición observados para validar modelos y apoyar respuesta a emergencias.

## Marco Teórico: Ecuaciones del Fenómeno

### Ecuación de Advección-Difusión Tridimensional

La evolución de la concentración (C(x,y,z,t)) se describe por
$$
\frac{\partial C}{\partial t} + \frac{\partial (uC)}{\partial x} + \frac{\partial (vC)}{\partial y} + \frac{\partial (wC)}{\partial z}
= \frac{\partial}{\partial x}\left(K_H \frac{\partial C}{\partial x}\right) + \frac{\partial}{\partial y}\left(K_H \frac{\partial C}{\partial y}\right) + \frac{\partial}{\partial z}\left(K_V \frac{\partial C}{\partial z}\right) + S - \lambda C
$$
con ((u,v,w)) las componentes del viento, (K_H) y (K_V) las difusividades, (S) la fuente y (\lambda) la tasa de remoción (decaimiento + deposición).

### Parametrización de la Difusión Turbulenta

Horizontal:
$$
K_H = C_K \cdot \Delta x^{4/3}
$$
con (\Delta x) la resolución espacial y (C_K) constante empírica.

Vertical (Monin–Obukhov, forma corregida):
$$
K_V(z) = \frac{\kappa,u_*,z}{\phi_m!\big(\tfrac{z}{L}\big)}
$$
donde (\kappa) es la constante de von Kármán, (u_*) velocidad de fricción, (z) altura, (L) longitud de Monin–Obukhov y (\phi_m) función de estabilidad.

### Procesos de Deposición

Deposición seca:
$$
F_d = v_d \cdot C(z_0)
$$
con (v_d) velocidad de deposición y (C(z_0)) concentración cerca de la superficie.

Deposición húmeda:
$$
\frac{\partial C}{\partial t}\bigg|_{wet} = -\Lambda \cdot C,\qquad \Lambda = a\cdot P^b
$$
donde (P) es la precipitación y (a,b) parámetros empíricos.

### Decaimiento Radiactivo

Evolución del número de átomos:
$$
\frac{dN}{dt} = -\lambda_r N,\qquad \lambda_r = \frac{\ln 2}{t_{1/2}}
$$
(p. ej. (t_{1/2}) para Cs-137 (\approx 30.17) años, para I-131 (\approx 8.02) días).

### Condiciones de Frontera e Iniciales

Superficie:
$$

* K_V\frac{\partial C}{\partial z}\bigg|_{z=0} = v_d, C(z=0)
  $$
  Condición inicial (inicio de emisiones):
  $$
  C(x,y,z,t=0) = C_0(x,y,z).
  $$

### Término Fuente de Emisión

Fuente localizada en ((x_0,y_0)) con distribución vertical (f(z)):
$$
S(x,y,z,t)=Q(t),\delta(x-x_0),\delta(y-y_0),f(z)
$$
donde (Q(t)) es la tasa temporal de emisión.

## Métodos y Algoritmos de Simulación

### Discretización Espacial y Temporal

Dominio hemisférico discretizado en malla horizontal (coordenadas esféricas: lat/lon) y vertical en coordenadas sigma o de presión. Resolución típica horizontal: (1\text{–}5^\circ) (refinamiento cerca de Chernobyl). Niveles verticales densos en la capa límite. Integración temporal con pasos adaptativos que respeten criterios de estabilidad.

### Esquemas de Advección

Se usan esquemas de alto orden (p. ej. Bott) o semi-Lagrangianos. En un esquema semi-Lagrangiano:
$$
C(x_i,t+\Delta t)=C(x_i - u\Delta t,t)
$$
requiriendo interpolación en posiciones no alineadas con la malla; permite pasos mayores respetando CFL efectivo.

### Tratamiento de la Difusión

Difusivos por diferencias finitas centradas (segundo orden). Ejemplo en (x):
$$
\frac{\partial}{\partial x}!\left(K_H\frac{\partial C}{\partial x}\right)\approx\frac{1}{\Delta x^2}\big[K_{i+1/2}(C_{i+1}-C_i)-K_{i-1/2}(C_i-C_{i-1})\big].
$$
Integración temporal implícita (Crank–Nicolson) para estabilidad incondicional o explícita con restricción:
$$
\Delta t \le \frac{(\Delta x)^2}{2K_H}.
$$

### Acoplamiento con Campos Meteorológicos

Los campos ((u,v,w)) provienen de reanálisis o pronósticos (6 h o mejores) y deben aproximar la conservación de masa:
$$
\frac{\partial u}{\partial x}+\frac{\partial v}{\partial y}+\frac{\partial w}{\partial z}=0.
$$
Si no se satisface, se aplica ajuste de masa (solución de una ecuación de Poisson para corrección).

### Integración Temporal y Fraccionamiento de Operadores

Se emplea splitting operator: en cada paso temporal se resuelven secuencialmente advección horizontal, advección vertical, difusión horizontal/vertical, y términos fuente/sumidero (emisiones, deposición, decaimiento). Facilita métodos especializados por proceso.

### Tratamiento de Condiciones de Frontera

Fronteras laterales con zonas de relajación para evitar reflexiones; frontera superior con condición de gradiente nulo que permite salida de material; superficie con balance de flujo y deposición como indicado.

## Metodología

### Escenarios de Simulación

Escenario base: periodo 26-abr a 9-may-1986 (fase principal de emisiones). Escenarios adicionales variando altura efectiva de emisión, perfil temporal (Q(t)), coeficientes de difusión y velocidades de deposición para análisis de sensibilidad y cuantificación de incertidumbre.

### Parámetros del Modelo

Parámetros relevantes:

* Cs-137 como trazador de largo plazo; emisión total estimada ~85 PBq (literatura).
* Altura efectiva de emisión: 200–1500 m (dependiente de flotabilidad).
* Velocidad de deposición seca (v_d): 0.001–0.01 m/s según superficie.
* Coeficientes (K_H) según resolución; (K_V) según parametrizaciones de capa límite.
* Perfil temporal de emisiones reconstruido históricamente (picos iniciales y declive).

### Datos Meteorológicos de Entrada

Campos tridimensionales de viento, temperatura, humedad y precipitación (reanálisis históricos) en intervalos compatibles (p. ej. 6 h). Requieren interpolación espacial y temporal para el paso del modelo; la precisión de estos campos es crítica para reproducir patrones de deposición.

### Diseño del Experimento Computacional

Fases: inicialización (concentraciones nulas), fase de emisión activa (10 días) con inyección según (Q(t)), y fase post-emisión (~20 días o más) para dispersión y deposición a distancia. Se implementan pruebas de verificación numérica (conservación de masa, análisis de estabilidad, comparaciones analíticas simplificadas).

### Criterios de Evaluación

Comparación con observaciones de deposición mediante métricas: correlación espacial, sesgo, error cuadrático medio normalizado (NRMSE), capacidad para reproducir dirección predominante del transporte, zonas de deposición intensa ligadas a precipitaciones, y atenuación con distancia desde la fuente.

Aquí tienes un informe para el aaaaah no te creas mathew, ahora a pedro le toca fajarse el resto del proyecto
