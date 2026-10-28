import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const RoadEdgeApp());
}

class RoadEdgeApp extends StatelessWidget {
  const RoadEdgeApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'RoadEdge-BR',
      theme: ThemeData.dark(),
      home: const RoadEdgeCameraScreen(),
    );
  }
}

class RoadEdgeCameraScreen extends StatefulWidget {
  const RoadEdgeCameraScreen({super.key});

  @override
  State<RoadEdgeCameraScreen> createState() =>
      _RoadEdgeCameraScreenState();
}

class _RoadEdgeCameraScreenState extends State<RoadEdgeCameraScreen> {
  final YOLOViewController _controller = YOLOViewController();

  static const int _totalAmostras = 1000;

  final List<double> _tempos = [];
  final List<double> _fpsValores = [];

  int _detections = 0;

  double? _processingTime;
  double? _fps;

  bool _benchmarkFinalizado = false;

  double _percentile(List<double> valoresOrdenados, double p) {
    if (valoresOrdenados.isEmpty) {
      return 0;
    }

    final position = (valoresOrdenados.length - 1) * p;
    final lower = position.floor();
    final upper = position.ceil();

    if (lower == upper) {
      return valoresOrdenados[lower];
    }

    final weight = position - lower;

    return valoresOrdenados[lower] * (1 - weight) +
        valoresOrdenados[upper] * weight;
  }

  void _registrarMetricas(dynamic metrics) {
    if (_benchmarkFinalizado) {
      return;
    }

final tempo = metrics.processingTimeMs.toDouble();
final fpsAtual = metrics.fps.toDouble();

if (tempo <= 0) {
  return;
}

_tempos.add(tempo);

if (fpsAtual > 0) {
  _fpsValores.add(fpsAtual);
}

    if (_tempos.length % 100 == 0) {
      debugPrint(
        'Benchmark câmera: ${_tempos.length}/$_totalAmostras',
      );
    }

    if (_tempos.length >= _totalAmostras) {
      _finalizarBenchmark();
    }
  }

  void _finalizarBenchmark() {
    if (_benchmarkFinalizado || _tempos.isEmpty) {
      return;
    }

    _benchmarkFinalizado = true;

    final temposOrdenados = List<double>.from(_tempos)..sort();

    final soma =
        _tempos.fold<double>(0, (anterior, atual) => anterior + atual);

    final media = soma / _tempos.length;

    final mediana = _percentile(temposOrdenados, 0.50);
    final p95 = _percentile(temposOrdenados, 0.95);
    final p99 = _percentile(temposOrdenados, 0.99);

    final minimo = temposOrdenados.first;
    final maximo = temposOrdenados.last;

    final fpsMedio = _fpsValores.isEmpty
        ? 0.0
        : _fpsValores.fold<double>(
              0,
              (anterior, atual) => anterior + atual,
            ) /
            _fpsValores.length;

    debugPrint('');
    debugPrint('==========================================');
    debugPrint('ROADEDGE-BR - BENCHMARK CÂMERA AO VIVO');
    debugPrint('==========================================');
    debugPrint('Modelo: YOLOv8n FP32 384x384');
    debugPrint('Backend: GPU');
    debugPrint('Confidence: 0.25');
    debugPrint('IoU: 0.70');
    debugPrint('Amostras: ${_tempos.length}');
    debugPrint('------------------------------------------');
    debugPrint('Média: ${media.toStringAsFixed(2)} ms');
    debugPrint('Mediana: ${mediana.toStringAsFixed(2)} ms');
    debugPrint('P95: ${p95.toStringAsFixed(2)} ms');
    debugPrint('P99: ${p99.toStringAsFixed(2)} ms');
    debugPrint('Mínimo: ${minimo.toStringAsFixed(2)} ms');
    debugPrint('Máximo: ${maximo.toStringAsFixed(2)} ms');
    debugPrint(
      'FPS médio reportado: ${fpsMedio.toStringAsFixed(2)}',
    );
    debugPrint('==========================================');
    debugPrint('');
  }

  @override
  void initState() {
    super.initState();

    WidgetsBinding.instance.addPostFrameCallback((_) async {
      try {
        await _controller.setConfidenceThreshold(0.25);
        await _controller.setIoUThreshold(0.70);
        await _controller.setShowOverlays(true);

        debugPrint('');
        debugPrint('===== ROADEDGE YOLO LIVE =====');
        debugPrint('Modelo: YOLOv8n FP32 384x384');
        debugPrint('Confidence: 0.25');
        debugPrint('IoU: 0.70');
        debugPrint('Benchmark: $_totalAmostras amostras');
        debugPrint('==============================');
        debugPrint('');
      } catch (e) {
        debugPrint('Erro ao configurar YOLOView: $e');
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text('RoadEdge-BR • Benchmark ao vivo'),
        centerTitle: true,
      ),
      body: Stack(
        children: [
          Positioned.fill(
            child: YOLOView(
              modelPath:
                  'assets/models/yolov8n_fp32_384.tflite',
              task: YOLOTask.detect,
              controller: _controller,
              useGpu: true,
              onResult: (results) {
                if (!mounted) {
                  return;
                }

                setState(() {
                  _detections = results.length;
                });
              },
              onPerformanceMetrics: (metrics) {
                _registrarMetricas(metrics);

                if (!mounted) {
                  return;
                }

                setState(() {
                  _processingTime = metrics.processingTimeMs;
                  _fps = metrics.fps;
                });
              },
            ),
          ),

          Positioned(
            left: 12,
            right: 12,
            bottom: 20,
            child: Container(
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                color: Colors.black.withValues(alpha: 0.75),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Text(
                    'YOLOv8n FP32 • 384 × 384 • GPU',
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 8),

                  Text('Detecções: $_detections'),

                  const SizedBox(height: 4),

                  Text(
                    _processingTime == null
                        ? 'Inferência: aguardando...'
                        : 'Inferência: '
                            '${_processingTime!.toStringAsFixed(2)} ms',
                  ),

                  const SizedBox(height: 4),

                  Text(
                    _fps == null
                        ? 'FPS: aguardando...'
                        : 'FPS: ${_fps!.toStringAsFixed(2)}',
                  ),

                  const SizedBox(height: 8),

                  Text(
                    _benchmarkFinalizado
                        ? 'Benchmark concluído: '
                            '${_tempos.length}/$_totalAmostras'
                        : 'Benchmark: '
                            '${_tempos.length}/$_totalAmostras',
                    style: TextStyle(
                      fontWeight: _benchmarkFinalizado
                          ? FontWeight.bold
                          : FontWeight.normal,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}