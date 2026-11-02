# Registro de correcciones respecto a la auditoría

Cada fila enlaza el hallazgo de la auditoría con el cambio realizado y la prueba que lo verifica.

| ID | Gravedad | Problema original | Corrección | Archivo(s) | Prueba |
|---|---|---|---|---|---|
| C1 | CRÍTICO | Ruido en banda base con varianza N0/4 por componente (3,01 dB de menos) → curva simulada desplazada ~3 dB | Varianza N0/2 (complejo) y N0/4 (real pasabanda), con derivación en el docstring | `core/channel.py` | `test_noise_calibration`, `test_ber_matches_theory` |
| C2 | CRÍTICO | LPF del demodulador cortaba en 0,5·Rs → ISI, EVM 0,24 sin ruido, piso de error | Corte [(1+α)/2 + 0,5]·Rs, dependiente de α | `core/modulator.py` | `test_noiseless_chain_has_no_errors` (EVM < 0,02) |
| A1 | ALTO | Mapeo no Gray en PAM 8/16, PSK 8/16, QAM 64/256 | `constelación[Gray(p)] = punto[p]` | `core/mapper.py` | `test_gray_adjacency_all_orders` |
| A2 | ALTO | Curva BER ignoraba el roll-off (fijo 0,35) | Curva con α del usuario | `core/simulation_runner.py` | `test_alpha_is_used_in_curve` |
| A3 | ALTO | Curva (banda base) y punto de operación (pasabanda) eran sistemas distintos | Un único enlace `CommLink` para ambos | `core/link.py` | `test_ber_matches_theory` |
| A4 | ALTO | Ojo antes del filtro casado y eje temporal con paso erróneo | Ojo a la salida del filtro casado; eje exacto −1…+1 T, muestreo en t = 0 | `plots/eye_diagram.py` | `TestEye` |
| A5 | ALTO | Potencia reportada = mitad de la real (÷2 duplicado) | Potencia sin factores ocultos, normalizada a Es/Ts, con teórica Es/2 | `metrics/signal_power.py` | `test_power_and_energy` |
| A6 | ALTO | Faltaba la tabla de entrega y su análisis/gráficas | Módulo de comparación, diálogo GUI, script y exportación | `analysis/`, `gui/comparison_dialog.py`, `scripts/` | `test_delivery_table` |
| M1 | MEDIO | BER teórica de 2-PSK con factor 2 de más | BPSK = Q(√(2γb)) | `core/ber_calculator.py` | `test_bpsk_equals_2pam` |
| M2 | MEDIO | Es constante = 1; ancho de banda solo normalizado; Eb no mostrado | Es teórica y medida, Eb, B en Hz, Rb, EVM | `metrics/`, `gui/controls_panel.py` | `test_power_and_energy`, `test_bandwidth_metrics` |
| M3 | MEDIO | Espectro "pasabanda RX" era la señal TX sin ruido; periodograma único | Se guarda la señal RX con ruido; PSD de Welch | `plots/spectrum.py`, `core/link.py` | inspección visual |
| M4 | MEDIO | Roll-off recortado en silencio; núcleo sin validación; BER mínima medible muy alta | Entradas de texto con mensajes; validación también en el núcleo; Monte Carlo hasta N errores | `utils/validators.py`, `gui/controls_panel.py`, `core/*` | `TestValidation` |
| B1 | BAJO | Rendimiento (bucles Python), código muerto, docstrings contradictorios, Python 3.9, colas del filtro truncadas, sin pruebas | Demapeo vectorizado, decisor por bloques, convolución FFT, relleno de símbolos, `from __future__ import annotations`, eliminación de código muerto, 29 pruebas, repositorio git | todo | `tests/` |
| RF1 | — | Resultados requerían pulsar Simular y bloqueaban hasta terminar | Dos etapas (punto inmediato + curva con progreso), actualización automática y cancelación | `gui/main_window.py` | prueba GUI offscreen |

Notas honestas:
* Verificado en Python 3.12; la compatibilidad con 3.9 se apoya en `from __future__ import annotations`
  pero no se ejecutó en 3.9.
* Con α = 0 el SRRC truncado a ±10 T conserva algo de ISI (EVM ≈ 0,1); es inherente a truncar la sinc.
* BER = 1e-6 y 1e-7 solo pueden obtenerse con la expresión teórica (no medibles por Monte Carlo razonable).

## Ronda 2 de la auditoría

| ID | Gravedad | Problema | Corrección | Archivo(s) | Prueba |
|---|---|---|---|---|---|
| H1 | ALTO | Con α pequeño el SRRC truncado a ±10 T dejaba ISI y un piso de error (16-PAM, α=0, 40 dB: BER 7e-2) | Span adaptativo según α (20/40/100) y aviso en la GUI para α<0,05. Con α=0 queda ISI residual inherente (BER 4e-3 en ese caso) | `utils/constants.py`, `core/link.py`, `utils/validators.py`, `gui/controls_panel.py` | `TestSmallRollOff` |
| H2 | MEDIO | El ruido se calibraba con la potencia medida en cada realización | Calibración con Es teórica de la constelación | `core/link.py` | `TestNoiseCalibrationTheoreticalEs` |
| H4 | MEDIO | Se afirmaba "desde cero" pero se usaban `np.convolve` y `np.fft` | FFT radix-2, convolución directa y por FFT propias; Welch sin `np.fft`; versiones polifásicas para rapidez | `utils/dsp.py`, `plots/spectrum.py`, `core/pulse_shaper.py` | `TestOwnDSP` |
| H5 | BAJO | El ojo de PAM/BPSK dibujaba la rama Q (solo ruido) | `show_q` explícito; la GUI no la dibuja con constelaciones reales | `plots/eye_diagram.py`, `gui/plots_panel.py` | `TestEyeQBranch` |
| H7 | BAJO | Definiciones de B y η sin justificar | Tabla de decisiones en `docs/TEORIA.md` §5 | `docs/TEORIA.md` | — |

Notas honestas:
* α = 0 sigue teniendo ISI residual (no se puede eliminar con un filtro finito).
* H3 (BER 1e-5…1e-7 no medibles por Monte Carlo) no se "corrige": es un límite físico del método y
  está explicado; esas filas de la tabla se obtienen con la expresión teórica.
* El código es algo más lento que con `np.convolve` (curva por defecto ≈ 6 s frente a ≈ 3 s) por usar
  convolución propia.
