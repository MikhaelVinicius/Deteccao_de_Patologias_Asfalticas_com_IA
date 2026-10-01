import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

void main() {
  runApp(const RoadEdgeApp());
}

class RoadEdgeApp extends StatelessWidget {
  const RoadEdgeApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'RoadEdge-BR',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.deepOrange,
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  String _status = 'Carregando modelo...';

  List<Map<String, dynamic>> _boxes = [];

  bool _processando = true;

  @override
  void initState() {
    super.initState();
    _executarTeste();
  }

  Future<void> _executarTeste() async {
    try {
      final yolo = YOLO(
        modelPath: 'assets/models/yolov8n_w8a16_calib1000.tflite',
        task: YOLOTask.detect,

        // O W8A16 apresentou incompatibilidade com o
        // delegate GPU no moto g14.
        useGpu: false,
      );

      await yolo.loadModel();

      if (mounted) {
        setState(() {
          _status = 'Modelo carregado. Processando imagem...';
        });
      }

      final ByteData data =
          await rootBundle.load('assets/images/pothole_test.png');

      final Uint8List imageBytes =
          data.buffer.asUint8List();

     // ============================================================
// BENCHMARK ROADEDGE-BR
// 3 warm-ups + 20 inferências medidas
// ============================================================

const int warmupRuns = 3;
const int benchmarkRuns = 20;

// ---------- WARM-UP ----------
debugPrint('===== WARM-UP =====');

for (int i = 0; i < warmupRuns; i++) {
  final stopwatch = Stopwatch()..start();

  await yolo.predict(
    imageBytes,
    confidenceThreshold: 0.25,
    iouThreshold: 0.70,
  );

  stopwatch.stop();

  debugPrint(
    'Warm-up ${i + 1}/$warmupRuns: '
    '${stopwatch.elapsedMilliseconds} ms',
  );
}

// ---------- BENCHMARK ----------
debugPrint('===== BENCHMARK MOTO G14 =====');

final List<double> temposMs = [];

Map<String, dynamic> resultado = {};

for (int i = 0; i < benchmarkRuns; i++) {
  final stopwatch = Stopwatch()..start();

  resultado = await yolo.predict(
    imageBytes,
    confidenceThreshold: 0.25,
    iouThreshold: 0.70,
  );

  stopwatch.stop();

  final double tempo =
      stopwatch.elapsedMicroseconds / 1000.0;

  temposMs.add(tempo);

  debugPrint(
    'Inferência ${i + 1}/$benchmarkRuns: '
    '${tempo.toStringAsFixed(2)} ms',
  );
}

// ---------- ESTATÍSTICAS ----------
final List<double> temposOrdenados =
    List<double>.from(temposMs)..sort();

final double media =
    temposMs.reduce((a, b) => a + b) /
        temposMs.length;

double mediana;

if (temposOrdenados.length.isOdd) {
  mediana =
      temposOrdenados[temposOrdenados.length ~/ 2];
} else {
  final int meio =
      temposOrdenados.length ~/ 2;

  mediana =
      (temposOrdenados[meio - 1] +
              temposOrdenados[meio]) /
          2;
}

double percentile(
  List<double> valores,
  double p,
) {
  final int index =
      ((valores.length - 1) * p).ceil();

  return valores[index];
}

final double p95 =
    percentile(temposOrdenados, 0.95);

final double p99 =
    percentile(temposOrdenados, 0.99);

final double minimo =
    temposOrdenados.first;

final double maximo =
    temposOrdenados.last;

final double fps =
    1000.0 / media;

debugPrint('');
debugPrint('===== RESULTADO BENCHMARK =====');
debugPrint('Dispositivo: moto g14');
debugPrint('Modelo: YOLOv8n W8A16');
debugPrint('Backend: CPU');
debugPrint('Warm-ups: $warmupRuns');
debugPrint('Inferências medidas: $benchmarkRuns');

debugPrint(
  'Média: ${media.toStringAsFixed(2)} ms',
);

debugPrint(
  'Mediana: ${mediana.toStringAsFixed(2)} ms',
);

debugPrint(
  'P95: ${p95.toStringAsFixed(2)} ms',
);

debugPrint(
  'P99: ${p99.toStringAsFixed(2)} ms',
);

debugPrint(
  'Mínimo: ${minimo.toStringAsFixed(2)} ms',
);

debugPrint(
  'Máximo: ${maximo.toStringAsFixed(2)} ms',
);

debugPrint(
  'FPS equivalente: ${fps.toStringAsFixed(2)}',
);

debugPrint('==============================');

      final List<dynamic> resultadoBoxes =
          (resultado['boxes'] as List<dynamic>?) ??
              <dynamic>[];

      final List<Map<String, dynamic>> boxes =
          resultadoBoxes
              .map(
                (box) => Map<String, dynamic>.from(
                  box as Map,
                ),
              )
              .toList();

      debugPrint('===== ROADEDGE RESULTADO =====');
      debugPrint('Detecções: ${boxes.length}');
      debugPrint('Boxes: $boxes');
      debugPrint('==============================');

      if (!mounted) return;

      setState(() {
        _boxes = boxes;
        _processando = false;

        if (boxes.isEmpty) {
          _status =
              'Inferência concluída — nenhuma detecção';
        } else {
          _status =
              'Inferência concluída com sucesso';
        }
      });
    } catch (e, stackTrace) {
      debugPrint('Erro RoadEdge: $e');
      debugPrint('$stackTrace');

      if (!mounted) return;

      setState(() {
        _processando = false;
        _status = 'Erro durante a inferência:\n$e';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('RoadEdge-BR'),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            AspectRatio(
              aspectRatio: 1.86,
              child: Stack(
                fit: StackFit.expand,
                children: [
                  Image.asset(
                    'assets/images/pothole_test.png',
                    fit: BoxFit.fill,
                  ),

                  if (!_processando)
                    CustomPaint(
                      painter: DetectionPainter(
                        boxes: _boxes,
                      ),
                    ),
                ],
              ),
            ),

            const SizedBox(height: 24),

            if (_processando)
              const CircularProgressIndicator(),

            const SizedBox(height: 20),

            Text(
              _status,
              textAlign: TextAlign.center,
              style: const TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),

            const SizedBox(height: 12),

            Text(
              'Detecções encontradas: ${_boxes.length}',
              style: const TextStyle(
                fontSize: 16,
              ),
            ),

            const SizedBox(height: 8),

            const Text(
              'Threshold de confiança: 0.25',
              style: TextStyle(
                fontSize: 14,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class DetectionPainter extends CustomPainter {
  final List<Map<String, dynamic>> boxes;

  DetectionPainter({
    required this.boxes,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final Paint boxPaint = Paint()
      ..color = Colors.red
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;

    final Paint backgroundPaint = Paint()
      ..color = Colors.red.withValues(
        alpha: 0.85,
      );

    for (final box in boxes) {
      final double x1 =
          (box['x1_norm'] as num).toDouble() *
              size.width;

      final double y1 =
          (box['y1_norm'] as num).toDouble() *
              size.height;

      final double x2 =
          (box['x2_norm'] as num).toDouble() *
              size.width;

      final double y2 =
          (box['y2_norm'] as num).toDouble() *
              size.height;

      final Rect rect = Rect.fromLTRB(
        x1,
        y1,
        x2,
        y2,
      );

      canvas.drawRect(
        rect,
        boxPaint,
      );

      final String className =
          box['className']?.toString() ??
              'pothole';

      final double confidence =
          (box['confidence'] as num)
              .toDouble();

      final String label =
          '$className ${(confidence * 100).toStringAsFixed(0)}%';

      final TextPainter textPainter =
          TextPainter(
        text: TextSpan(
          text: label,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 12,
            fontWeight: FontWeight.bold,
          ),
        ),
        textDirection: TextDirection.ltr,
      );

      textPainter.layout();

      double labelTop =
          y1 - textPainter.height - 6;

      if (labelTop < 0) {
        labelTop = y1;
      }

      final Rect labelBackground =
          Rect.fromLTWH(
        x1,
        labelTop,
        textPainter.width + 8,
        textPainter.height + 6,
      );

      canvas.drawRect(
        labelBackground,
        backgroundPaint,
      );

      textPainter.paint(
        canvas,
        Offset(
          x1 + 4,
          labelTop + 3,
        ),
      );
    }
  }

  @override
  bool shouldRepaint(
    covariant DetectionPainter oldDelegate,
  ) {
    return oldDelegate.boxes != boxes;
  }
}