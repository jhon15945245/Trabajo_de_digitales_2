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
