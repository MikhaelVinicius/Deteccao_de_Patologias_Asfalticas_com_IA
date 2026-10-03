import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  final cameras = await availableCameras();

  runApp(
    RoadEdgeApp(cameras: cameras),
  );
}

class RoadEdgeApp extends StatelessWidget {
  final List<CameraDescription> cameras;

  const RoadEdgeApp({
    super.key,
    required this.cameras,
  });

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'RoadEdge-BR',
      theme: ThemeData.dark(),
      home: CameraScreen(cameras: cameras),
    );
  }
}

class CameraScreen extends StatefulWidget {
  final List<CameraDescription> cameras;

  const CameraScreen({
    super.key,
    required this.cameras,
  });

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _controller;

  bool _inicializando = true;
  String? _erro;

  @override
  void initState() {
    super.initState();
    _inicializarCamera();
  }

  Future<void> _inicializarCamera() async {
    if (widget.cameras.isEmpty) {
      setState(() {
        _erro = 'Nenhuma câmera encontrada.';
        _inicializando = false;
      });
      return;
    }

    try {
      // Procura primeiro a câmera traseira.
      CameraDescription cameraSelecionada = widget.cameras.first;

      for (final camera in widget.cameras) {
        if (camera.lensDirection == CameraLensDirection.back) {
          cameraSelecionada = camera;
          break;
        }
      }

      final controller = CameraController(
        cameraSelecionada,
        ResolutionPreset.medium,
        enableAudio: false,
      );

      await controller.initialize();

      if (!mounted) {
        await controller.dispose();
        return;
      }

      setState(() {
        _controller = controller;
        _inicializando = false;
      });

      debugPrint('===== ROADEDGE CÂMERA =====');
      debugPrint('Câmeras encontradas: ${widget.cameras.length}');
      debugPrint('Câmera utilizada: ${cameraSelecionada.name}');
      debugPrint('Direção: ${cameraSelecionada.lensDirection}');
      debugPrint('Preview inicializado com sucesso.');
      debugPrint('===========================');
    } catch (e) {
      debugPrint('Erro ao inicializar câmera: $e');

      if (!mounted) {
        return;
      }

      setState(() {
        _erro = e.toString();
        _inicializando = false;
      });
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text('RoadEdge-BR • Teste de câmera'),
        centerTitle: true,
      ),
      body: _construirConteudo(),
    );
  }

  Widget _construirConteudo() {
    if (_inicializando) {
      return const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('Inicializando câmera...'),
          ],
        ),
      );
    }

    if (_erro != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(
                Icons.error_outline,
                size: 64,
              ),
              const SizedBox(height: 16),
              const Text(
                'Não foi possível inicializar a câmera.',
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 12),
              Text(
                _erro!,
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      );
    }

    final controller = _controller;

    if (controller == null || !controller.value.isInitialized) {
      return const Center(
        child: Text('Câmera não disponível.'),
      );
    }

    return Column(
      children: [
        Expanded(
          child: Center(
            child: AspectRatio(
              aspectRatio: controller.value.aspectRatio,
              child: CameraPreview(controller),
            ),
          ),
        ),
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(16),
          child: const Column(
            children: [
              Text(
                'Câmera ativa',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
              SizedBox(height: 6),
              Text(
                'Nesta etapa o YOLO ainda não está executando.',
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      ],
    );
  }
}