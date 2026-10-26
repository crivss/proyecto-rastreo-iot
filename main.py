
import numpy as np

from métodos_numéricos import (
    metodo_secante,
    trilateracion_gauss_seidel
)


print("=" * 55)
print(" PROYECTO DE RASTREO IoT - ESPOCH")
print("=" * 55)


# ==========================================
# FASE 1: AUTOCALIBRACION CON SECANTE
# ==========================================

distancia_calibracion = 5.0
potencia_transmitida = 0.0
perdida_referencia = 40.0
potencia_recibida = -60.08


def funcion_calibracion(n):
    return (
        10 * n * np.log10(distancia_calibracion)
        + potencia_recibida
        - potencia_transmitida
        + perdida_referencia
    )


n_estimado, historial_secante, convergio_secante = metodo_secante(
    funcion_calibracion,
    n0=2.0,
    n1=4.0,
    tolerancia=1e-4,
    max_iteraciones=50
)

print("\n1. AUTOCALIBRACION DEL EXPONENTE")
print("-" * 55)

for fila in historial_secante:
    print(
        f"Iteracion {fila['iteracion']}: "
        f"n = {fila['n']:.6f}, "
        f"error = {fila['error_absoluto']:.6f}"
    )

print(f"Exponente estimado: {n_estimado:.6f}")
print(f"Convergencia: {'Si' if convergio_secante else 'No'}")


# ==========================================
# FASE 2: SIMULACION DE SEÑALES RSSI
# ==========================================

anclas = np.array([
    [0.0, 0.0],
    [10.0, 0.0],
    [0.0, 10.0],
    [10.0, 10.0]
])

# Posicion real usada solo para evaluar la simulacion
posicion_real = np.array([4.0, 7.5])

distancia_referencia = 1.0
ruido_estandar = 1.0

# Distancias reales entre las anclas y el dispositivo
distancias_reales = np.linalg.norm(
    anclas - posicion_real,
    axis=1
)

# Ruido reproducible
np.random.seed(42)

ruido_rssi = np.random.normal(
    loc=0.0,
    scale=ruido_estandar,
    size=len(anclas)
)

# Simular las señales recibidas
rssi = (
    potencia_transmitida
    - perdida_referencia
    - 10 * n_estimado
    * np.log10(distancias_reales / distancia_referencia)
    + ruido_rssi
)

# Convertir las señales RSSI en distancias estimadas
distancias_estimadas = distancia_referencia * 10 ** (
    (potencia_transmitida - perdida_referencia - rssi)
    / (10 * n_estimado)
)

print("\n2. ADQUISICION SIMULADA DE RSSI")
print("-" * 55)
print(f"{'Ancla':<10}{'RSSI (dBm)':<18}{'Distancia (m)':<18}")

for i in range(len(anclas)):
    print(
        f"{i + 1:<10}"
        f"{rssi[i]:<18.4f}"
        f"{distancias_estimadas[i]:<18.4f}"
    )


# ==========================================
# FASE 3 Y 4: CONDICION Y LOCALIZACION
# ==========================================

posicion_estimada, historial_gs, convergio_gs, condicion = (
    trilateracion_gauss_seidel(
        anclas,
        distancias_estimadas,
        semilla=(5.0, 5.0),
        tolerancia=1e-5,
        max_iteraciones=1000
    )
)

error_absoluto = np.linalg.norm(
    posicion_real - posicion_estimada
)

magnitud_real = np.linalg.norm(posicion_real)

if magnitud_real > 0:
    error_relativo = error_absoluto / magnitud_real * 100
else:
    error_relativo = None

print("\n3. ANALISIS DE CONDICION")
print("-" * 55)
print(f"Numero de condicion kappa(M): {condicion:.6f}")

print("\n4. LOCALIZACION CON GAUSS-SEIDEL")
print("-" * 70)
print(f"Iteraciones realizadas: {len(historial_gs)}")
print(f"Convergencia: {'Si' if convergio_gs else 'No'}")

print("\nRESULTADOS FINALES")
print("-" * 55)
print(f"Posicion real:     X = {posicion_real[0]:.4f} m, "
      f"Y = {posicion_real[1]:.4f} m")
print(f"Posicion estimada: X = {posicion_estimada[0]:.4f} m, "
      f"Y = {posicion_estimada[1]:.4f} m")
print(f"Error absoluto: {error_absoluto:.6f} m")

if error_relativo is not None:
    print(f"Error relativo porcentual: {error_relativo:.4f} %")

print("\nSimulacion finalizada.")

