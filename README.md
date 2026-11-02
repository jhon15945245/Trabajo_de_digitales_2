# Simulador de un sistema de comunicación digital (M-PAM / M-QAM / M-PSK)

Universidad del Cauca — Facultad de Ingeniería Electrónica y Telecomunicaciones.
Simulación de la cadena completa: mapeo → conformación (SRRC) → modulación IQ en
pasabanda → canal AWGN → demodulación → filtro casado → decisor → demapeo.

## Instalación y ejecución

```bash
pip install -r requirements.txt     # PyQt5, matplotlib, numpy
python main.py                      # interfaz gráfica
```

Pruebas automáticas (solo biblioteca estándar) y tabla de entrega:

```bash
python -m unittest discover -s tests -v
python scripts/generar_tabla_entrega.py --simular --salida resultados
```

## Estructura (sistema modular)

| Carpeta | Contenido | Bloque del sistema |
|---|---|---|
| `core/mapper.py` | Constelaciones PAM/QAM/PSK con codificación Gray | Transmisor |
| `core/pulse_shaper.py` | Filtro SRRC (conformador y filtro casado) | Transmisor / Receptor |
| `core/modulator.py` | Modulador y demodulador IQ coherente | Transmisor / Receptor |
| `core/channel.py` | Canal AWGN calibrado por Eb/N0 | Canal |
| `core/decision.py` | Decisor de mínima distancia y demapeo | Receptor |
| `core/link.py` | Enlace completo TX → canal → RX (un solo camino para todo) | Integración |
| `core/ber_calculator.py` | BER teórica y simulada | Análisis |
| `core/simulation_runner.py` | Punto de operación y curva Monte Carlo | Orquestador |
| `metrics/` | Eficiencia espectral, Es/Eb, ancho de banda, potencia | RF4 |
| `plots/` | Ojo, constelación, BER, espectro, comparaciones | RF3 |
| `analysis/` | Tabla de entrega (ΔEb/N0), análisis y exportación | Entrega 4 |
| `gui/` | Interfaz PyQt5 | RF1, RF2, RNF1 |
| `utils/` | Constantes, validadores, DSP propio (convolución/FFT) | — |
| `tests/` | Pruebas automáticas | — |
| `docs/` | Teoría usada (`TEORIA.md`) y registro de correcciones | — |

## Uso de la interfaz

* Elija esquema, orden M (4, 16, 64, 256 en QAM; 2, 4, 8, 16 en PAM y PSK) y roll-off 0 ≤ α ≤ 1.
  Con α < 0,05 aparece un aviso: un filtro finito no puede ser Nyquist exacto cuando α → 0 y queda ISI
  residual (el span del SRRC crece automáticamente: 20 / 40 / 100 símbolos según α; ver `docs/TEORIA.md`).
* Los campos se validan al instante: un valor no permitido (α = 1.7, M = 8 en QAM, texto, vacío…)
  muestra un mensaje bajo el campo y un diálogo al pulsar *Simular*; la aplicación sigue funcionando.
* Con **Actualizar automáticamente** (por defecto) los resultados se actualizan al cambiar un
  parámetro: primero ojo, constelación, espectro y métricas (inmediato) y luego la curva BER.
* **Comparar 4PAM / 4QAM / 4PSK** abre la tabla de entrega con análisis y gráficas, verificación
  por simulación y exportación (CSV, Markdown, PNG, análisis).

## Restricción RNF2 (sin funciones de alto nivel)

No se usa `scipy`, `commpy`, toolboxes ni modulaciones prediseñadas.  Todo (constelaciones,
SRRC, modulador, demodulador, LPF, canal, decisor, **convolución, FFT radix-2**, Welch, Q(x),
inversión de Q) está implementado en el proyecto (`utils/dsp.py`).  De numpy solo se usan arreglos
(indexación, suma, producto, `exp`, `sin`, `cos`) y el generador aleatorio gaussiano/uniforme; de la
biblioteca estándar, `math.erfc` para Q(x).  **No se usa `np.fft` ni `np.convolve`**; las pruebas
(`tests/`) los emplean únicamente como referencia para validar las implementaciones propias.

## Reproducibilidad (RNF3)

Todo el azar sale de `numpy.random.default_rng` con semillas fijas derivadas de `RANDOM_SEED = 42`
(bits y ruido de cada bloque Monte Carlo).  Ejecutar varias veces da resultados idénticos.

## Control de versiones

El repositorio git contiene el estado original entregado a auditoría y las correcciones posteriores
(`git log`).  Continúe con commits propios al hacer cambios (configure `git config user.name/email`).
