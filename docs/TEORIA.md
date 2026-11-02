# Fundamento teórico implementado (para la sustentación)

Notación: Rs = tasa de símbolo, Ts = 1/Rs, sps = muestras por símbolo (fs = sps·Rs), k = log2 M,
α = roll-off, γb = Eb/N0 (lineal).  Conversión: γb = 10^(Eb/N0[dB]/10).

## 1. Cadena y archivos

| Etapa | Operación | Archivo |
|---|---|---|
| 1 | bits → símbolos (Gray), Es = 1 | `core/mapper.py` |
| 2 | sobremuestreo ×sps y SRRC | `core/pulse_shaper.py` |
| 3 | x(t) = I·cos(2πfc t) − Q·sin(2πfc t), fc = fs/4 | `core/modulator.py` |
| 4 | ruido AWGN según Eb/N0 | `core/channel.py` |
| 5 | mezcla 2cos / −2sin + LPF | `core/modulator.py` |
| 6 | filtro casado SRRC, muestreo cada sps | `core/pulse_shaper.py` |
| 7 | decisión por mínima distancia, demapeo | `core/decision.py` |

## 2. Definiciones (rúbrica: "Definición")

| Magnitud | Fórmula | Dónde |
|---|---|---|
| Es (teórica) | (1/M)·Σ\|sᵢ\|² = 1 | `metrics/symbol_energy.py` |
| Es (medida) | P_bb·sps, con P_bb = E\|s[n]\|² y SRRC de energía unitaria | idem |
| Eb | Es / log2 M | idem |
| N0 | Eb / γb | `core/channel.py` |
| Ruido (complejo) | σ² = N0/2 por componente | `core/channel.py` |
| Ruido (real pasabanda) | σ_r² = N0/4 (el demodulador multiplica por 2 → 2σ_r² = N0/2) | idem |
| Potencia pasabanda | P_pb = E[x²] = P_bb/2 = Es/(2·sps) por muestra; Es/2 por Ts | `metrics/signal_power.py` |
| Eficiencia espectral | η = log2(M)/(1+α) bit/s/Hz | `metrics/spectral_efficiency.py` |
| Ancho de banda | B = Rs·(1+α) (pasabanda, nulo a nulo) | `metrics/bandwidth.py` |

**¿Por qué N0/2 por componente?**  Tras el filtro casado de energía unitaria (que no cambia la varianza
de un ruido blanco) cada componente debe tener varianza N0/2, que es la densidad bilateral del ruido
AWGN.  La distancia mínima entre símbolos fija entonces Q(d_min/√(2N0)).

## 3. BER teórica (Gray)

* M-PAM:  Pb ≈ [2(M−1)/(M·k)]·Q(√(6kγb/(M²−1)))
* M-QAM:  Pb ≈ [4(1−1/√M)/k]·Q(√(3kγb/(M−1)))
* M-PSK:  BPSK: Q(√(2γb));  M ≥ 4: Pb ≈ (2/k)·Q(√(2kγb)·sin(π/M))

4-QAM y 4-PSK son la misma constelación rotada 45°: misma Pb.
4-PAM frente a 4-QAM: argumentos de Q √(0,8γb) frente a √(2γb) → diferencia asintótica 10·log10(2,5) = 3,98 dB.

## 4. Preguntas probables y respuestas cortas

| Pregunta | Respuesta |
|---|---|
| ¿Por qué SRRC en TX y RX? | La cascada es coseno alzado (ISI nula en nTs) y RX es el filtro casado (máxima SNR). |
| ¿Qué cambia si aumento α? | Más ancho de banda (1+α)·Rs y menor eficiencia, pero colas más cortas, ojo más abierto y menos sensibilidad al instante de muestreo. La BER ideal no cambia (ISI nula). |
| ¿Por qué se rellena con símbolos nulos? | Para no truncar las colas del SRRC en el primer y último símbolo. |
| ¿Por qué el LPF corta en [(1+α)/2 + 0,5]·Rs? | Debe pasar todo el espectro de la señal y rechazar el término en 2fc. |
| ¿Por qué la BER simulada de 1e-6 no aparece? | Exigiría ~1e8–1e9 bits; un punto sin errores solo da una cota BER < 1/bits. |
| ¿Qué es Gray y por qué importa? | Vecinos difieren en 1 bit: un error de símbolo al vecino cuesta 1 bit; las fórmulas de BER lo asumen. |
| ¿Por qué Eb/N0 y no SNR? | Permite comparar esquemas con distinto k a igual energía por bit. |
| ¿Cómo se pasa de dB a lineal? | γ = 10^(dB/10); 3,01 dB ≈ factor 2. |

## 5. Límites conocidos y decisiones de diseño (para la sustentación)

| Tema | Decisión | Justificación |
|---|---|---|
| Span del SRRC | 20 símbolos si α ≥ 0,2; 40 si α ≥ 0,05; 100 si α < 0,05 | La cola del SRRC decae como 1/t cuando α → 0; truncarla deja ISI. EVM sin ruido: α=0,05 → 0,005 (span 40); α=0,10 → 0,001 |
| α = 0 | Se permite, con ISI residual acotada (EVM ≈ 0,045) | Un filtro de longitud finita no puede ser Nyquist con α=0. La BER simulada queda por encima de la teórica (≈ ×1,2 en 4/16-QAM, mayor en órdenes altos). Ventanear el SRRC empeoró el resultado (se probó con Hann) |
| Calibración de Eb/N0 | Con Es **teórica** de la constelación (Es = 1) | Así Eb/N0 no depende de la muestra aleatoria. Con SRRC de energía unitaria: P_bb = Es/sps y P_pb = P_bb/2 |
| Ancho de banda | B = Rs·(1+α), señal PASABANDA (doble banda lateral, nulo a nulo) | El enunciado pide espectro en pasabanda. En banda base el espectro ocupa ±B/2 |
| Eficiencia espectral | η = log2(M)/(1+α) | Igual para PAM, QAM y PSK modulados en pasabanda con una portadora. Con M-PAM en banda base (sin portadora) η valdría el doble |
| Monte Carlo | BER mínima medible ≈ 1/(bits simulados) | BER 1e-6 y 1e-7 exigirían ≈ 1e8–1e9 bits; ahí se usa la expresión teórica, validada donde sí es medible |
| FFT / convolución | Implementadas en `utils/dsp.py` (radix-2 y directa) | RNF2. Directa para filtros ≤ 400 coeficientes (más rápida), FFT para más |
