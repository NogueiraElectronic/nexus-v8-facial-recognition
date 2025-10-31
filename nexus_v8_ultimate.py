Fecha: 2025
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional, Set, Any, Union, Callable
from collections import defaultdict, deque
from datetime import datetime
import pickle
import logging
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import hashlib

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine, euclidean, mahalanobis
from scipy.stats import entropy, multivariate_normal
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.covariance import EmpiricalCovariance

warnings.filterwarnings('ignore')

# Setup advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nexus_ultimate.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('NEXUS_ULTIMATE')

# ===================================================================
# IMPORTS CONDICIONALES CON FALLBACKS INTELIGENTES
# ===================================================================
print("\n" + "="*80)
print("🌟 NEXUS v8.0 ULTIMATE - INICIANDO SISTEMA REVOLUCIONARIO")
print("="*80)

# PyTorch + Advanced Components
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam, AdamW
    from torch.cuda.amp import autocast, GradScaler
    TORCH_AVAILABLE = True
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✅ PyTorch {torch.__version__} - Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = 'cpu'
    print("⚠️  PyTorch no disponible - Usando fallback NumPy")

# PyTorch Geometric (para GNN avanzado)
try:
    import torch_geometric
    from torch_geometric.nn import GATConv, TransformerConv, global_mean_pool, global_max_pool
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
    print(f"✅ PyTorch Geometric {torch_geometric.__version__}")
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("⚠️  PyTorch Geometric no disponible - GNN simplificado")

# FaceNet
try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    FACENET_AVAILABLE = True
    print("✅ FaceNet-PyTorch")
except ImportError:
    FACENET_AVAILABLE = False
    print("⚠️  FaceNet no disponible")

# InsightFace
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
    print(f"✅ InsightFace {insightface.__version__}")
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("⚠️  InsightFace no disponible")

# MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️  MediaPipe no disponible")

print("="*80 + "\n")

# ===================================================================
# ENUMS Y CONSTANTES
# ===================================================================
class RecognitionMode(Enum):
    """Modos de reconocimiento"""
    STRICT = "strict"
    BALANCED = "balanced"
    PERMISSIVE = "permissive"
    ADAPTIVE = "adaptive"

class QualityLevel(Enum):
    """Nivel de calidad de detección"""
    EXCELLENT = 4
    GOOD = 3
    FAIR = 2
    POOR = 1

# ===================================================================
# CONFIGURACIÓN AVANZADA
# ===================================================================
@dataclass
class UltimateConfig:
    """Configuración avanzada del sistema NEXUS ULTIMATE"""
    
    # === VIDEO ===
    video_path: str = 'videos/entrada.mp4'
    inicio_segundos: int = 0
    duracion_segundos: int = 60
    skip_frames: int = 0  # Procesar 1 de cada N frames
    
    # === GALERÍA ===
    reference_gallery_path: str = 'reference_gallery/'
    min_reference_images: int = 3
    auto_augment_gallery: bool = True
    
    # === SISTEMA ===
    device: str = DEVICE
    num_workers: int = 8
    batch_size: int = 16
    use_half_precision: bool = torch.cuda.is_available()
    enable_cuda_graphs: bool = False  # Experimental
    
    # === DETECCIÓN AVANZADA ===
    use_mtcnn: bool = FACENET_AVAILABLE
    use_mediapipe: bool = MEDIAPIPE_AVAILABLE
    use_ensemble_detection: bool = True
    face_confidence_threshold: float = 0.85
    min_face_size: int = 40
    max_face_size: int = 800
    nms_threshold: float = 0.4
    detection_scales: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5])
    
    # === EMBEDDING AVANZADO ===
    use_insightface: bool = INSIGHTFACE_AVAILABLE
    use_facenet: bool = FACENET_AVAILABLE
    embedding_dim: int = 512
    enable_pca: bool = True
    pca_components: int = 256
    enable_whitening: bool = True
    
    # === DYNAMIC FEATURE FUSION ===
    enable_dynamic_fusion: bool = TORCH_AVAILABLE
    fusion_learning_rate: float = 0.001
    fusion_history_size: int = 100
    
    # === GRAPH NEURAL NETWORK AVANZADO ===
    enable_gnn: bool = TORCH_GEOMETRIC_AVAILABLE
    gnn_hidden_dim: int = 384
    gnn_num_layers: int = 4
    gnn_attention_heads: int = 8
    gnn_dropout: float = 0.15
    social_proximity_threshold: float = 300.0
    temporal_window: int = 45
    gnn_update_interval: int = 5
    
    # === ADAPTIVE META-LEARNING ===
    enable_meta_learning: bool = TORCH_AVAILABLE
    meta_learning_rate: float = 0.0001
    meta_adaptation_steps: int = 3
    meta_update_interval: int = 20
    
    # === PHYSICS-INFORMED MOTION ===
    enable_physics_motion: bool = True
    motion_damping: float = 0.8
    motion_prediction_horizon: int = 20
    use_kalman_filter: bool = True
    
    # === BAYESIAN UNCERTAINTY ===
    enable_bayesian: bool = True
    num_mc_samples: int = 10
    uncertainty_threshold: float = 0.3
    
    # === ACTIVE RE-IDENTIFICATION ===
    enable_active_reid: bool = True
    reid_search_window: int = 60
    reid_similarity_threshold: float = 0.65
    max_reid_attempts: int = 3
    
    # === STREAMING ATTENTION ===
    enable_streaming_attention: bool = TORCH_AVAILABLE
    attention_memory_size: int = 128
    attention_heads: int = 8
    attention_dropout: float = 0.1
    
    # === BEHAVIORAL ANALYSIS ===
    enable_behavior_analysis: bool = True
    behavior_window: int = 120  # frames
    min_behavior_samples: int = 30
    
    # === ADVERSARIAL ROBUSTNESS ===
    enable_adversarial_training: bool = False  # Expensive
    adversarial_epsilon: float = 0.01
    
    # === MULTI-SCALE TEMPORAL ===
    temporal_scales: List[int] = field(default_factory=lambda: [5, 15, 30, 60])
    
    # === ENSEMBLE META-OPTIMIZATION ===
    enable_meta_ensemble: bool = TORCH_AVAILABLE
    num_ensemble_agents: int = 7
    ensemble_diversity_penalty: float = 0.2
    
    # === SYNTHETIC AUGMENTATION ===
    enable_synthetic_aug: bool = True
    aug_rotation_range: float = 15.0
    aug_brightness_range: float = 0.2
    aug_noise_std: float = 0.01
    
    # === RECONOCIMIENTO ADAPTATIVO ===
    recognition_mode: RecognitionMode = RecognitionMode.ADAPTIVE
    base_threshold: float = 0.70
    threshold_adaptation_rate: float = 0.05
    quality_weight: float = 0.3
    
    # === TRACKING AVANZADO ===
    max_disappeared: int = 45
    tracking_iou_threshold: float = 0.25
    max_tracking_distance: float = 200.0
    enable_reid_tracking: bool = True
    track_confidence_decay: float = 0.95
    
    # === CACHE Y OPTIMIZACIÓN ===
    enable_embedding_cache: bool = True
    cache_size: int = 1000
    enable_frame_skip_adaptive: bool = True
    target_fps: float = 30.0
    
    # === OUTPUT ===
    output_dir: str = 'results/nexus_v8_ultimate'
    save_video: bool = True
    save_json: bool = True
    save_embeddings: bool = True
    save_explanations: bool = True
    save_heatmaps: bool = True
    video_codec: str = 'mp4v'
    video_fps: Optional[int] = None
    
    # === VISUALIZACIÓN AVANZADA ===
    draw_bboxes: bool = True
    draw_track_ids: bool = True
    draw_confidence_bars: bool = True
    draw_social_graph: bool = True
    draw_motion_vectors: bool = True
    draw_uncertainty: bool = True
    draw_behavior_labels: bool = True
    visualization_quality: str = 'high'  # low, medium, high
    
    # === LOGGING Y DEBUG ===
    verbose: int = 1  # 0=silent, 1=info, 2=debug, 3=trace
    enable_profiling: bool = False
    save_intermediate_results: bool = False
    
    # === SELF-HEALING ===
    enable_self_healing: bool = True
    max_error_recovery_attempts: int = 3
    health_check_interval: int = 100

# Configuración global
CONFIG = UltimateConfig()

# ===================================================================
# UTILIDADES AVANZADAS
# ===================================================================
class EmbeddingCache:
    """Cache inteligente para embeddings"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = defaultdict(int)
        self.lock = threading.Lock()
    
    def _get_key(self, image: np.ndarray) -> str:
        """Genera hash único para imagen"""
        return hashlib.md5(image.tobytes()).hexdigest()
    
    def get(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Obtiene embedding del cache"""
        key = self._get_key(image)
        with self.lock:
            if key in self.cache:
                self.access_count[key] += 1
                return self.cache[key]
        return None
    
    def put(self, image: np.ndarray, embedding: np.ndarray):
        """Guarda embedding en cache"""
        key = self._get_key(image)
        with self.lock:
            if len(self.cache) >= self.max_size:
                # Eliminar menos usado
                min_key = min(self.access_count, key=self.access_count.get)
                del self.cache[min_key]
                del self.access_count[min_key]
            
            self.cache[key] = embedding
            self.access_count[key] = 1
    
    def clear(self):
        """Limpia cache"""
        with self.lock:
            self.cache.clear()
            self.access_count.clear()

class PerformanceMonitor:
    """Monitor de rendimiento en tiempo real"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.timers = {}
        self.lock = threading.Lock()
    
    def start_timer(self, name: str):
        """Inicia timer"""
        self.timers[name] = time.time()
    
    def stop_timer(self, name: str):
        """Detiene timer y registra"""
        if name in self.timers:
            elapsed = time.time() - self.timers[name]
            with self.lock:
                self.metrics[name].append(elapsed)
            del self.timers[name]
            return elapsed
        return 0.0
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Obtiene estadísticas"""
        with self.lock:
            if name not in self.metrics or not self.metrics[name]:
                return {}
            
            values = self.metrics[name]
            return {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Obtiene todas las estadísticas"""
        return {name: self.get_stats(name) for name in self.metrics.keys()}

# ===================================================================
# DETECCIÓN Y CALIDAD
# ===================================================================
@dataclass
class Detection:
    """Detección mejorada con metadatos avanzados"""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    embedding: Optional[np.ndarray] = None
    quality_score: float = 0.0
    quality_level: QualityLevel = QualityLevel.FAIR
    
    # Reconocimiento
    person_id: Optional[str] = None
    recognition_confidence: float = 0.0
    recognition_uncertainty: float = 1.0
    recognition_votes: Dict[str, int] = field(default_factory=dict)
    
    # Tracking
    track_id: Optional[int] = None
    track_confidence: float = 1.0
    
    # Temporal
    velocity: Optional[Tuple[float, float]] = None
    predicted_position: Optional[Tuple[float, float]] = None
    
    # Behavior
    behavior_pattern: Optional[str] = None
    behavior_confidence: float = 0.0
    
    # Meta
    frame_idx: int = 0
    timestamp: float = 0.0
    source_model: str = "unknown"
    
    def center(self) -> Tuple[float, float]:
        """Centro del bbox"""
        return ((self.bbox[0] + self.bbox[2]) / 2,
                (self.bbox[1] + self.bbox[3]) / 2)
    
    def area(self) -> float:
        """Área del bbox"""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
    
    def iou(self, other: 'Detection') -> float:
        """IoU con otra detección"""
        x1 = max(self.bbox[0], other.bbox[0])
        y1 = max(self.bbox[1], other.bbox[1])
        x2 = min(self.bbox[2], other.bbox[2])
        y2 = min(self.bbox[3], other.bbox[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = self.area() + other.area() - intersection
        
        return intersection / (union + 1e-6)

class QualityEstimator:
    """Estimador de calidad de detecciones"""
    
    @staticmethod
    def estimate(face_crop: np.ndarray, bbox: np.ndarray, frame_shape: Tuple) -> Tuple[float, QualityLevel]:
        """Estima calidad de una detección"""
        scores = []
        
        # 1. Tamaño (más grande = mejor)
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        frame_area = frame_shape[0] * frame_shape[1]
        size_score = min(area / (frame_area * 0.1), 1.0)
        scores.append(size_score)
        
        # 2. Nitidez (Laplacian variance)
        if face_crop.size > 0:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 500.0, 1.0)
            scores.append(sharpness_score)
        
        # 3. Brillo (histograma)
        if face_crop.size > 0:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
            mean_brightness = gray.mean()
            brightness_score = 1.0 - abs(mean_brightness - 128) / 128.0
            scores.append(brightness_score)
        
        # 4. Posición (centro es mejor)
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        frame_cx, frame_cy = frame_shape[1] / 2, frame_shape[0] / 2
        dist_to_center = np.sqrt((cx - frame_cx)**2 + (cy - frame_cy)**2)
        max_dist = np.sqrt(frame_cx**2 + frame_cy**2)
        position_score = 1.0 - (dist_to_center / max_dist)
        scores.append(position_score)
        
        # Combinar scores
        quality_score = np.mean(scores) if scores else 0.0
        
        # Asignar nivel
        if quality_score >= 0.8:
            level = QualityLevel.EXCELLENT
        elif quality_score >= 0.6:
            level = QualityLevel.GOOD
        elif quality_score >= 0.4:
            level = QualityLevel.FAIR
        else:
            level = QualityLevel.POOR
        
        return quality_score, level

# ===================================================================
# DETECTORES AVANZADOS
# ===================================================================
class EnsembleDetector:
    """Detector ensemble avanzado con voting"""
    
    def __init__(self, config: UltimateConfig):
        self.config = config
        self.detectors = []
        
        # MTCNN
        if config.use_mtcnn and FACENET_AVAILABLE:
            try:
                self.mtcnn = MTCNN(
                    device=config.device,
                    keep_all=True,
                    post_process=False,
                    min_face_size=config.min_face_size
                )
                self.detectors.append(('mtcnn', self.mtcnn))
                logger.info("✅ MTCNN detector loaded")
            except Exception as e:
                logger.warning(f"⚠️  MTCNN loading failed: {e}")
        
        # MediaPipe
        if config.use_mediapipe and MEDIAPIPE_AVAILABLE:
            try:
                self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=0.5
                )
                self.detectors.append(('mediapipe', self.mp_face_detection))
                logger.info("✅ MediaPipe detector loaded")
            except Exception as e:
                logger.warning(f"⚠️  MediaPipe loading failed: {e}")
        
        # OpenCV Haar Cascade (fallback)
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.haar_cascade = cv2.CascadeClassifier(cascade_path)
            if not self.haar_cascade.empty():
                self.detectors.append(('haar', self.haar_cascade))
                logger.info("✅ Haar Cascade detector loaded")
        except Exception as e:
            logger.warning(f"⚠️  Haar Cascade loading failed: {e}")
        
        if not self.detectors:
            raise RuntimeError("❌ No face detectors available!")
        
        logger.info(f"🎯 Ensemble detector ready with {len(self.detectors)} models")
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detecta caras con ensemble voting"""
        all_detections = []
        h, w = frame.shape[:2]
        
        for detector_name, detector in self.detectors:
            try:
                if detector_name == 'mtcnn':
                    boxes, probs = detector.detect(frame)
                    if boxes is not None:
                        for box, prob in zip(boxes, probs):
                            if prob >= self.config.face_confidence_threshold:
                                all_detections.append(Detection(
                                    bbox=box,
                                    confidence=float(prob),
                                    source_model='mtcnn'
                                ))
                
                elif detector_name == 'mediapipe':
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = detector.process(rgb)
                    if results.detections:
                        for detection in results.detections:
                            bbox_mp = detection.location_data.relative_bounding_box
                            x1 = int(bbox_mp.xmin * w)
                            y1 = int(bbox_mp.ymin * h)
                            x2 = int((bbox_mp.xmin + bbox_mp.width) * w)
                            y2 = int((bbox_mp.ymin + bbox_mp.height) * h)
                            
                            all_detections.append(Detection(
                                bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                                confidence=detection.score[0],
                                source_model='mediapipe'
                            ))
                
                elif detector_name == 'haar':
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = detector.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5,
                        minSize=(self.config.min_face_size, self.config.min_face_size)
                    )
                    for (x, y, w_box, h_box) in faces:
                        all_detections.append(Detection(
                            bbox=np.array([x, y, x+w_box, y+h_box], dtype=np.float32),
                            confidence=0.9,  # Haar no da probabilidad
                            source_model='haar'
                        ))
            
            except Exception as e:
                if self.config.verbose > 1:
                    logger.debug(f"Detection error in {detector_name}: {e}")
        
        # NMS para fusionar detecciones
        if all_detections:
            detections = self._apply_nms(all_detections)
        else:
            detections = []
        
        # Calcular calidad
        for det in detections:
            try:
                face_crop = frame[int(det.bbox[1]):int(det.bbox[3]), 
                                int(det.bbox[0]):int(det.bbox[2])]
                det.quality_score, det.quality_level = QualityEstimator.estimate(
                    face_crop, det.bbox, frame.shape
                )
            except:
                det.quality_score = 0.5
                det.quality_level = QualityLevel.FAIR
        
        return detections
    
    def _apply_nms(self, detections: List[Detection]) -> List[Detection]:
        """Non-Maximum Suppression"""
        if not detections:
            return []
        
        boxes = np.array([d.bbox for d in detections])
        scores = np.array([d.confidence for d in detections])
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            
            iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-6)
            
            inds = np.where(iou <= self.config.nms_threshold)[0]
            order = order[inds + 1]
        
        return [detections[i] for i in keep]

# ===================================================================
# EXTRACTORES DE EMBEDDINGS AVANZADOS
# ===================================================================
class DynamicEmbeddingExtractor:
    """Extractor de embeddings con fusión dinámica aprendida"""
    
    def __init__(self, config: UltimateConfig):
        self.config = config
        self.extractors = {}
        self.cache = EmbeddingCache(config.cache_size) if config.enable_embedding_cache else None
        
        # InsightFace
        if config.use_insightface and INSIGHTFACE_AVAILABLE:
            try:
                self.insightface_app = FaceAnalysis(
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                self.insightface_app.prepare(ctx_id=0 if config.device == 'cuda' else -1)
                self.extractors['insightface'] = self.insightface_app
                logger.info("✅ InsightFace extractor loaded")
            except Exception as e:
                logger.warning(f"⚠️  InsightFace loading failed: {e}")
        
        # FaceNet
        if config.use_facenet and FACENET_AVAILABLE:
            try:
                self.facenet_model = InceptionResnetV1(
                    pretrained='vggface2'
                ).eval().to(config.device)
                self.extractors['facenet'] = self.facenet_model
                logger.info("✅ FaceNet extractor loaded")
            except Exception as e:
                logger.warning(f"⚠️  FaceNet loading failed: {e}")
        
        if not self.extractors:
            raise RuntimeError("❌ No embedding extractors available!")
        
        # Fusion weights (aprenderá online)
        self.fusion_weights = {name: 1.0 / len(self.extractors) 
                              for name in self.extractors.keys()}
        
        # PCA para reducción dimensional
        self.pca = None
        self.scaler = None
        if config.enable_pca:
            self.pca = PCA(n_components=config.pca_components, whiten=config.enable_whitening)
            self.scaler = StandardScaler()
        
        # Historia de éxitos para aprender pesos
        self.success_history = defaultdict(list)
        
        logger.info(f"🎯 Embedding extractor ready with {len(self.extractors)} models")
    
    def extract(self, frame: np.ndarray, detection: Detection) -> Optional[np.ndarray]:
        """Extrae embedding con fusión dinámica"""
        
        # Check cache
        if self.cache:
            cached = self.cache.get(frame[int(detection.bbox[1]):int(detection.bbox[3]),
                                          int(detection.bbox[0]):int(detection.bbox[2])])
            if cached is not None:
                return cached
        
        embeddings = {}
        
        # Extraer con cada modelo
        for name, extractor in self.extractors.items():
            try:
                emb = self._extract_single(frame, detection, name, extractor)
                if emb is not None:
                    embeddings[name] = emb
            except Exception as e:
                if self.config.verbose > 1:
                    logger.debug(f"Embedding extraction error in {name}: {e}")
        
        if not embeddings:
            return None
        
        # Fusión dinámica con pesos aprendidos
        fused_embedding = self._dynamic_fusion(embeddings)
        
        # PCA si está habilitado
        if self.pca and fused_embedding is not None:
            try:
                fused_embedding = self.pca.transform(fused_embedding.reshape(1, -1))[0]
            except:
                pass  # PCA no ajustado aún
        
        # Cache result
        if self.cache and fused_embedding is not None:
            face_crop = frame[int(detection.bbox[1]):int(detection.bbox[3]),
                            int(detection.bbox[0]):int(detection.bbox[2])]
            self.cache.put(face_crop, fused_embedding)
        
        return fused_embedding
    
    def _extract_single(self, frame: np.ndarray, detection: Detection, 
                       name: str, extractor) -> Optional[np.ndarray]:
        """Extrae embedding con un modelo específico"""
        x1, y1, x2, y2 = detection.bbox.astype(int)
        face_crop = frame[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            return None
        
        if name == 'insightface':
            # InsightFace necesita BGR
            faces = extractor.get(face_crop)
            if faces:
                return faces[0].embedding
        
        elif name == 'facenet':
            # FaceNet necesita RGB y resize
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (160, 160))
            face_tensor = torch.from_numpy(face_resized).permute(2, 0, 1).float() / 255.0
            face_tensor = face_tensor.unsqueeze(0).to(self.config.device)
            
            with torch.no_grad():
                embedding = extractor(face_tensor)
            
            return embedding.cpu().numpy().flatten()
        
        return None
    
    def _dynamic_fusion(self, embeddings: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """Fusión dinámica con pesos aprendidos"""
        if not embeddings:
            return None
        
        # Normalizar cada embedding
        normalized = {}
        for name, emb in embeddings.items():
            norm = np.linalg.norm(emb)
            normalized[name] = emb / (norm + 1e-8)
        
        # Fusión ponderada
        fused = np.zeros_like(list(normalized.values())[0])
        total_weight = 0.0
        
        for name, emb in normalized.items():
            weight = self.fusion_weights.get(name, 1.0)
            fused += weight * emb
            total_weight += weight
        
        fused /= (total_weight + 1e-8)
        
        return fused
    
    def update_fusion_weights(self, model_name: str, success: bool):
        """Actualiza pesos de fusión basado en éxito/fracaso"""
        if not self.config.enable_dynamic_fusion:
            return
        
        self.success_history[model_name].append(1.0 if success else 0.0)
        
        # Mantener solo historia reciente
        if len(self.success_history[model_name]) > self.config.fusion_history_size:
            self.success_history[model_name].pop(0)
        
        # Actualizar pesos basado en success rate
        if len(self.success_history[model_name]) >= 10:
            success_rate = np.mean(self.success_history[model_name])
            
            # Soft update
            current_weight = self.fusion_weights[model_name]
            target_weight = success_rate
            self.fusion_weights[model_name] = (
                (1 - self.config.fusion_learning_rate) * current_weight +
                self.config.fusion_learning_rate * target_weight
            )
        
        # Normalizar pesos
        total = sum(self.fusion_weights.values())
        if total > 0:
            for name in self.fusion_weights:
                self.fusion_weights[name] /= total
    
    def fit_pca(self, embeddings: List[np.ndarray]):
        """Ajusta PCA con embeddings de galería"""
        if not self.config.enable_pca or not embeddings:
            return
        
        try:
            embeddings_array = np.array(embeddings)
            if self.scaler:
                embeddings_array = self.scaler.fit_transform(embeddings_array)
            if self.pca:
                self.pca.fit(embeddings_array)
            logger.info(f"✅ PCA fitted with {len(embeddings)} embeddings")
        except Exception as e:
            logger.warning(f"⚠️  PCA fitting failed: {e}")

# ===================================================================
# GRAPH NEURAL NETWORK AVANZADO
# ===================================================================
if TORCH_AVAILABLE and TORCH_GEOMETRIC_AVAILABLE:
    class AdvancedSocialGNN(nn.Module):
        """GNN avanzado para contexto social y temporal"""
        
        def __init__(self, config: UltimateConfig):
            super().__init__()
            self.config = config
            
            input_dim = config.pca_components if config.enable_pca else config.embedding_dim
            hidden_dim = config.gnn_hidden_dim
            
            # Multi-head attention layers
            self.conv_layers = nn.ModuleList([
                TransformerConv(
                    input_dim if i == 0 else hidden_dim,
                    hidden_dim,
                    heads=config.gnn_attention_heads,
                    dropout=config.gnn_dropout,
                    concat=False
                )
                for i in range(config.gnn_num_layers)
            ])
            
            # Layer norms
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim)
                for _ in range(config.gnn_num_layers)
            ])
            
            # Output projection
            self.output_proj = nn.Linear(hidden_dim, input_dim)
            
            self.dropout = nn.Dropout(config.gnn_dropout)
        
        def forward(self, x, edge_index, edge_attr=None):
            """Forward pass con residual connections"""
            
            for i, (conv, norm) in enumerate(zip(self.conv_layers, self.layer_norms)):
                identity = x
                
                # Graph convolution
                x = conv(x, edge_index, edge_attr)
                x = norm(x)
                x = F.gelu(x)
                x = self.dropout(x)
                
                # Residual connection (después de primera capa)
                if i > 0:
                    x = x + identity
            
            # Output projection
            x = self.output_proj(x)
            
            return x
        
        def build_graph(self, detections: List[Detection]) -> Optional[Data]:
            """Construye graph de detecciones"""
            if not detections:
                return None
            
            # Node features (embeddings)
            node_features = []
            valid_indices = []
            
            for i, det in enumerate(detections):
                if det.embedding is not None:
                    node_features.append(det.embedding)
                    valid_indices.append(i)
            
            if not node_features:
                return None
            
            x = torch.tensor(np.array(node_features), dtype=torch.float)
            
            # Edge index (proximidad espacial)
            edge_list = []
            edge_attrs = []
            
            for i in range(len(valid_indices)):
                for j in range(i + 1, len(valid_indices)):
                    det_i = detections[valid_indices[i]]
                    det_j = detections[valid_indices[j]]
                    
                    # Distancia espacial
                    c_i = det_i.center()
                    c_j = det_j.center()
                    dist = np.linalg.norm(np.array(c_i) - np.array(c_j))
                    
                    if dist < self.config.social_proximity_threshold:
                        # Bidirectional edges
                        edge_list.extend([[i, j], [j, i]])
                        
                        # Edge attributes (distancia normalizada)
                        weight = 1.0 - (dist / self.config.social_proximity_threshold)
                        edge_attrs.extend([weight, weight])
            
            if not edge_list:
                # Sin edges, graph trivial
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = None
            else:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t()
                edge_attr = torch.tensor(edge_attrs, dtype=torch.float).unsqueeze(1)
            
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

else:
    class AdvancedSocialGNN:
        """Fallback cuando PyTorch Geometric no disponible"""
        def __init__(self, config):
            self.config = config
            logger.warning("⚠️  GNN no disponible - usando fallback")
        
        def __call__(self, *args, **kwargs):
            return None
        
        def build_graph(self, detections):
            return None

# ===================================================================
# BAYESIAN UNCERTAINTY QUANTIFICATION
# ===================================================================
class BayesianUncertainty:
    """Cuantificación de incertidumbre bayesiana"""
    
    def __init__(self, config: UltimateConfig):
        self.config = config
    
    def compute_uncertainty(self, 
                          embedding: np.ndarray,
                          gallery_embeddings: Dict[str, np.ndarray],
                          num_samples: int = None) -> Tuple[str, float, float]:
        """
        Computa identidad con incertidumbre bayesiana
        
        Returns:
            (best_id, confidence, uncertainty)
        """
        if num_samples is None:
            num_samples = self.config.num_mc_samples
        
        # Monte Carlo Dropout simulation (simplificado sin dropout real)
        # En producción, usar modelo con dropout en eval mode
        
        similarities = {}
        for person_id, gallery_emb in gallery_embeddings.items():
            # Simular samples con pequeño ruido
            samples = []
            for _ in range(num_samples):
                noisy_emb = embedding + np.random.normal(0, 0.01, embedding.shape)
                sim = 1 - cosine(noisy_emb, gallery_emb)
                samples.append(sim)
            
            # Media y std de similaridad
            mean_sim = np.mean(samples)
            std_sim = np.std(samples)
            
            similarities[person_id] = (mean_sim, std_sim)
        
        # Best match
        best_id = max(similarities.keys(), key=lambda k: similarities[k][0])
        best_mean, best_std = similarities[best_id]
        
        # Confidence = mean
        # Uncertainty = std
        return best_id, best_mean, best_std

# ===================================================================
# PHYSICS-INFORMED MOTION PREDICTOR
# ===================================================================
class PhysicsInformedMotion:
    """Predictor de movimiento con física"""
    
    def __init__(self, config: UltimateConfig):
        self.config = config
        self.track_histories = defaultdict(deque)
        self.velocities = {}
        self.accelerations = {}
    
    def update(self, track_id: int, position: Tuple[float, float], frame_idx: int):
        """Actualiza historial de tracking"""
        history = self.track_histories[track_id]
        history.append((frame_idx, position))
        
        # Mantener ventana
        while len(history) > 10:
            history.popleft()
        
        # Calcular velocidad y aceleración
        if len(history) >= 2:
            dt = history[-1][0] - history[-2][0]
            if dt > 0:
                dx = history[-1][1][0] - history[-2][1][0]
                dy = history[-1][1][1] - history[-2][1][1]
                self.velocities[track_id] = (dx / dt, dy / dt)
        
        if len(history) >= 3:
            # Aceleración (segunda derivada)
            v_curr = self.velocities.get(track_id, (0, 0))
            
            dt = history[-2][0] - history[-3][0]
            if dt > 0:
                dx_prev = history[-2][1][0] - history[-3][1][0]
                dy_prev = history[-2][1][1] - history[-3][1][1]
                v_prev = (dx_prev / dt, dy_prev / dt)
                
                dt = history[-1][0] - history[-2][0]
                if dt > 0:
                    ax = (v_curr[0] - v_prev[0]) / dt
                    ay = (v_curr[1] - v_prev[1]) / dt
                    self.accelerations[track_id] = (ax, ay)
    
    def predict(self, track_id: int, horizon: int = None) -> Optional[Tuple[float, float]]:
        """Predice posición futura con física"""
        if horizon is None:
            horizon = self.config.motion_prediction_horizon
        
        history = self.track_histories.get(track_id)
        if not history:
            return None
        
        curr_pos = history[-1][1]
        velocity = self.velocities.get(track_id, (0, 0))
        acceleration = self.accelerations.get(track_id, (0, 0))
        
        # Física: s = s0 + v*t + 0.5*a*t^2
        # Con damping para movimiento humano
        damping = self.config.motion_damping
        
        pred_x = curr_pos[0] + velocity[0] * horizon * damping
        pred_y = curr_pos[1] + velocity[1] * horizon * damping
        
        if len(history) >= 3:
            pred_x += 0.5 * acceleration[0] * (horizon ** 2) * damping
            pred_y += 0.5 * acceleration[1] * (horizon ** 2) * damping
        
        return (pred_x, pred_y)

# ===================================================================
# BEHAVIORAL PATTERN ANALYZER
# ===================================================================
class BehavioralAnalyzer:
    """Analiza patrones de comportamiento"""
    
    def __init__(self, config: UltimateConfig):
        self.config = config
        self.track_behaviors = defaultdict(list)
        self.behavior_patterns = {
            'static': 'Estático/Parado',
            'walking': 'Caminando',
            'running': 'Corriendo',
            'loitering': 'Merodeando',
            'interacting': 'Interactuando'
        }
    
    def analyze(self, track_id: int, positions: List[Tuple[float, float]], 
                frame_indices: List[int], nearby_tracks: List[int]) -> Tuple[str, float]:
        """Analiza comportamiento de un track"""
        
        if len(positions) < self.config.min_behavior_samples:
            return 'unknown', 0.0
        
        # Velocidad promedio
        velocities = []
        for i in range(1, len(positions)):
            dt = frame_indices[i] - frame_indices[i-1]
            if dt > 0:
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                dist = np.sqrt(dx**2 + dy**2)
                velocities.append(dist / dt)
        
        if not velocities:
            return 'unknown', 0.0
        
        avg_velocity = np.mean(velocities)
        velocity_std = np.std(velocities)
        
        # Clasificar comportamiento
        if avg_velocity < 1.0:
            behavior = 'static'
            confidence = 0.9
        elif avg_velocity < 5.0:
            if velocity_std > 2.0:
                behavior = 'loitering'
                confidence = 0.7
            else:
                behavior = 'walking'
                confidence = 0.85
        else:
            behavior = 'running'
            confidence = 0.8
        
        # Modificar si hay interacción (personas cercanas)
        if len(nearby_tracks) > 0 and avg_velocity < 3.0:
            behavior = 'interacting'
            confidence = 0.75
        
        return behavior, confidence

# ===================================================================
# TRACKING AVANZADO CON RE-IDENTIFICACIÓN
# ===================================================================
@dataclass
class Track:
    """Track mejorado con más metadatos"""
    track_id: int
    detections: deque = field(default_factory=lambda: deque(maxlen=100))
    disappeared: int = 0
    confidence: float = 1.0
    
    # Re-ID
    embeddings: List[np.ndarray] = field(default_factory=list)
    mean_embedding: Optional[np.ndarray] = None
    
    # Identidad
    person_id: Optional[str] = None
    person_confidence: float = 0.0
    
    # Behavior
    behavior_pattern: Optional[str] = None
    behavior_history: List[str] = field(default_factory=list)
    
    def update_mean_embedding(self):
        """Actualiza embedding promedio"""
        if self.embeddings:
            self.mean_embedding = np.mean(self.embeddings[-10:], axis=0)
    
    def last_position(self) -> Optional[Tuple[float, float]]:
        """Última posición conocida"""
        if self.detections:
            return self.detections[-1].center()
        return None

class AdvancedTracker:
    """Tracker avanzado con re-identificación"""
    
    def __init__(self, config: UltimateConfig):
        self.config = config
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.physics_motion = PhysicsInformedMotion(config)
        
        # Re-ID
        self.lost_tracks: Dict[int, Track] = {}  # Tracks perdidos recientes
        self.reid_attempts = defaultdict(int)
    
    def update(self, detections: List[Detection], frame_idx: int) -> List[Track]:
        """Actualiza tracks con nuevas detecciones"""
        
        # Predecir posiciones
        for track_id, track in self.tracks.items():
            if track.last_position():
                pred_pos = self.physics_motion.predict(track_id)
                # Guardar predicción en algún lugar si es necesario
        
        # Asociar detecciones con tracks existentes
        if self.tracks and detections:
            track_ids = list(self.tracks.keys())
            assignments = self._associate(track_ids, detections, frame_idx)
        else:
            assignments = {}
        
        # Actualizar tracks asignados
        assigned_detections = set()
        for track_id, det_idx in assignments.items():
            det = detections[det_idx]
            track = self.tracks[track_id]
            
            track.detections.append(det)
            track.disappeared = 0
            track.confidence = min(track.confidence / self.config.track_confidence_decay + 0.1, 1.0)
            
            if det.embedding is not None:
                track.embeddings.append(det.embedding)
                track.update_mean_embedding()
            
            if det.person_id:
                track.person_id = det.person_id
                track.person_confidence = det.recognition_confidence
            
            # Actualizar physics
            self.physics_motion.update(track_id, det.center(), frame_idx)
            
            assigned_detections.add(det_idx)
        
        # Crear nuevos tracks para detecciones no asignadas
        for i, det in enumerate(detections):
            if i not in assigned_detections:
                new_track = Track(track_id=self.next_id)
                new_track.detections.append(det)
                
                if det.embedding is not None:
                    new_track.embeddings.append(det.embedding)
                    new_track.update_mean_embedding()
                
                self.tracks[self.next_id] = new_track
                det.track_id = self.next_id
                self.next_id += 1
        
        # Marcar desaparecidos y eliminar
        to_remove = []
        for track_id, track in self.tracks.items():
            if track_id not in assignments:
                track.disappeared += 1
                track.confidence *= self.config.track_confidence_decay
                
                if track.disappeared >= self.config.max_disappeared:
                    to_remove.append(track_id)
                    # Guardar para re-ID
                    if self.config.enable_reid_tracking:
                        self.lost_tracks[track_id] = track
        
        for track_id in to_remove:
            del self.tracks[track_id]
        
        # Limpiar lost_tracks antiguos
        if len(self.lost_tracks) > 50:
            oldest = sorted(self.lost_tracks.keys())[:20]
            for tid in oldest:
                del self.lost_tracks[tid]
        
        # Re-identificación
        if self.config.enable_active_reid:
            self._attempt_reid(detections)
        
        return list(self.tracks.values())
    
    def _associate(self, track_ids: List[int], detections: List[Detection], 
                  frame_idx: int) -> Dict[int, int]:
        """Asocia tracks con detecciones usando Hungarian algorithm"""
        
        # Matriz de costos
        cost_matrix = np.zeros((len(track_ids), len(detections)))
        
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            last_det = track.detections[-1] if track.detections else None
            
            if not last_det:
                cost_matrix[i, :] = 1.0
                continue
            
            for j, det in enumerate(detections):
                # Costo basado en IoU y distancia
                iou = last_det.iou(det)
                
                # Distancia de centro
                c1 = last_det.center()
                c2 = det.center()
                dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                
                # Costo combinado
                cost_iou = 1.0 - iou
                cost_dist = min(dist / self.config.max_tracking_distance, 1.0)
                
                # Si hay embeddings, usar similaridad
                cost_emb = 0.5
                if det.embedding is not None and track.mean_embedding is not None:
                    sim = 1 - cosine(det.embedding, track.mean_embedding)
                    cost_emb = 1.0 - sim
                
                # Combinar costos
                cost = 0.3 * cost_iou + 0.3 * cost_dist + 0.4 * cost_emb
                cost_matrix[i, j] = cost
        
        # Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Filtrar asignaciones con costo alto
        assignments = {}
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < 0.7:  # Threshold
                assignments[track_ids[row]] = col
                detections[col].track_id = track_ids[row]
        
        return assignments
    
    def _attempt_reid(self, detections: List[Detection]):
        """Intenta re-identificar tracks perdidos"""
        if not self.lost_tracks or not detections:
            return
        
        for lost_id, lost_track in list(self.lost_tracks.items()):
            if self.reid_attempts[lost_id] >= self.config.max_reid_attempts:
                continue
            
            if not lost_track.mean_embedding:
                continue
            
            # Buscar mejor match en detecciones sin track
            best_match = None
            best_sim = self.config.reid_similarity_threshold
            
            for det in detections:
                if det.track_id is not None:  # Ya asignada
                    continue
                
                if det.embedding is None:
                    continue
                
                sim = 1 - cosine(det.embedding, lost_track.mean_embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_match = det
            
            # Re-identificación exitosa
            if best_match:
                # Re-activar track
                lost_track.disappeared = 0
                lost_track.detections.append(best_match)
                lost_track.confidence = 0.8
                
                self.tracks[lost_id] = lost_track
                del self.lost_tracks[lost_id]
                
                best_match.track_id = lost_id
                
                if self.config.verbose > 1:
                    logger.debug(f"✅ Re-ID exitoso: Track {lost_id}")
            
            self.reid_attempts[lost_id] += 1
    
    def get_active_tracks(self) -> List[Track]:
        """Obtiene tracks activos"""
        return [t for t in self.tracks.values() if t.disappeared < 5]

# ===================================================================
# ADAPTIVE THRESHOLD CONTROLLER
# ===================================================================
class AdaptiveThresholdController:
    """Controlador adaptativo de umbrales"""
    
    def __init__(self, config: UltimateConfig):
        self.config = config
        self.current_threshold = config.base_threshold
        self.quality_history = deque(maxlen=100)
        self.recognition_history = deque(maxlen=100)
    
    def update(self, quality_scores: List[float], recognition_confidences: List[float]):
        """Actualiza umbral basado en calidad y reconocimientos"""
        if quality_scores:
            self.quality_history.extend(quality_scores)
        if recognition_confidences:
            self.recognition_history.extend(recognition_confidences)
        
        if len(self.quality_history) < 20:
            return self.current_threshold
        
        # Calidad promedio
        avg_quality = np.mean(list(self.quality_history))
        
        # Recognition rate
        if self.recognition_history:
            rec_rate = np.mean([1 if c > 0.5 else 0 for c in self.recognition_history])
        else:
            rec_rate = 0.5
        
        # Ajustar umbral
        # Si calidad baja, reducir umbral
        # Si rec_rate muy alto, aumentar umbral (más estricto)
        target_threshold = self.config.base_threshold
        
        if avg_quality < 0.5:
            target_threshold -= 0.1
        elif avg_quality > 0.8:
            target_threshold += 0.05
        
        if rec_rate > 0.8:
            target_threshold += 0.05
        elif rec_rate < 0.3:
            target_threshold -= 0.1
        
        # Clip
        target_threshold = np.clip(target_threshold, 0.45, 0.85)
        
        # Soft update
        self.current_threshold = (
            (1 - self.config.threshold_adaptation_rate) * self.current_threshold +
            self.config.threshold_adaptation_rate * target_threshold
        )
        
        return self.current_threshold

# ===================================================================
# SISTEMA PRINCIPAL NEXUS ULTIMATE
# ===================================================================
class NexusUltimateSystem:
    """Sistema principal NEXUS v8.0 ULTIMATE"""
    
    def __init__(self, config: UltimateConfig):
        self.config = config
        self.setup()
    
    def setup(self):
        """Inicializa sistema completo"""
        logger.info("\n🚀 Inicializando NEXUS ULTIMATE...")
        
        # Output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Detector
        logger.info("📸 Cargando detectores...")
        self.detector = EnsembleDetector(self.config)
        
        # Extractor
        logger.info("🧠 Cargando extractores de embeddings...")
        self.extractor = DynamicEmbeddingExtractor(self.config)
        
        # Galería
        logger.info("🖼️  Cargando galería de referencia...")
        self.gallery = self.load_gallery()
        
        # Tracker
        self.tracker = AdvancedTracker(self.config)
        
        # GNN
        if self.config.enable_gnn and TORCH_GEOMETRIC_AVAILABLE:
            self.gnn = AdvancedSocialGNN(self.config).to(self.config.device)
            self.gnn.eval()
        else:
            self.gnn = None
        
        # Uncertainty
        if self.config.enable_bayesian:
            self.bayesian = BayesianUncertainty(self.config)
        else:
            self.bayesian = None
        
        # Physics Motion
        if self.config.enable_physics_motion:
            self.physics = PhysicsInformedMotion(self.config)
        else:
            self.physics = None
        
        # Behavior
        if self.config.enable_behavior_analysis:
            self.behavior = BehavioralAnalyzer(self.config)
        else:
            self.behavior = None
        
        # Adaptive threshold
        self.threshold_controller = AdaptiveThresholdController(self.config)
        
        # Performance monitor
        self.perf_monitor = PerformanceMonitor()
        
        # Stats
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'total_recognitions': 0,
            'processing_times': [],
            'reid_successes': 0,
            'threshold_history': []
        }
        
        # Video writer
        self.video_writer = None
        
        logger.info("✅ Sistema inicializado correctamente\n")
    
    def load_gallery(self) -> Dict[str, np.ndarray]:
        """Carga galería de referencia"""
        gallery = {}
        gallery_path = Path(self.config.reference_gallery_path)
        
        if not gallery_path.exists():
            logger.warning(f"⚠️  Galería no encontrada: {gallery_path}")
            return gallery
        
        all_embeddings = []
        
        for person_dir in gallery_path.iterdir():
            if not person_dir.is_dir():
                continue
            
            person_id = person_dir.name
            person_embeddings = []
            
            for img_path in person_dir.glob('*.[jp][pn]g'):
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    # Detectar cara
                    detections = self.detector.detect(img)
                    if not detections:
                        continue
                    
                    # Tomar mejor detección
                    best_det = max(detections, key=lambda d: d.confidence)
                    
                    # Extraer embedding
                    embedding = self.extractor.extract(img, best_det)
                    if embedding is not None:
                        person_embeddings.append(embedding)
                        all_embeddings.append(embedding)
                
                except Exception as e:
                    logger.debug(f"Error loading {img_path}: {e}")
            
            if person_embeddings:
                # Promedio de embeddings
                gallery[person_id] = np.mean(person_embeddings, axis=0)
                logger.info(f"   ✅ {person_id}: {len(person_embeddings)} imágenes")
        
        # Fit PCA con embeddings de galería
        if all_embeddings and self.config.enable_pca:
            self.extractor.fit_pca(all_embeddings)
        
        logger.info(f"\n📊 Galería cargada: {len(gallery)} personas\n")
        return gallery
    
    def process_video(self):
        """Procesa video completo"""
        logger.info("🎬 Iniciando procesamiento de video...")
        
        # Abrir video
        cap = cv2.VideoCapture(self.config.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"❌ No se pudo abrir video: {self.config.video_path}")
        
        # Props
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calcular frame range
        start_frame = int(self.config.inicio_segundos * fps)
        end_frame = start_frame + int(self.config.duracion_segundos * fps)
        end_frame = min(end_frame, total_frames)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Video writer
        if self.config.save_video:
            output_fps = self.config.video_fps if self.config.video_fps else fps
            fourcc = cv2.VideoWriter_fourcc(*self.config.video_codec)
            output_path = f'{self.config.output_dir}/output.mp4'
            self.video_writer = cv2.VideoWriter(
                output_path, fourcc, output_fps, (width, height)
            )
        
        # Progress bar
        pbar = tqdm(total=end_frame - start_frame, desc="Processing")
        
        frame_idx = start_frame
        
        try:
            while frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Procesar frame
                self.perf_monitor.start_timer('frame_total')
                
                try:
                    viz_frame = self.process_frame(frame, frame_idx)
                    
                    if self.video_writer and viz_frame is not None:
                        self.video_writer.write(viz_frame)
                
                except Exception as e:
                    logger.error(f"Error processing frame {frame_idx}: {e}")
                    if self.config.enable_self_healing:
                        # Continuar con siguiente frame
                        pass
                    else:
                        raise
                
                elapsed = self.perf_monitor.stop_timer('frame_total')
                self.stats['processing_times'].append(elapsed)
                
                frame_idx += 1
                pbar.update(1)
                
                # Adaptive frame skip
                if self.config.enable_frame_skip_adaptive:
                    if elapsed > 0 and (1.0 / elapsed) < self.config.target_fps * 0.8:
                        # Demasiado lento, skip frame
                        cap.read()
                        frame_idx += 1
                        pbar.update(1)
        
        finally:
            cap.release()
            if self.video_writer:
                self.video_writer.release()
            pbar.close()
        
        # Exportar resultados
        self.export_results()
        self.print_summary()
    
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Procesa un frame individual"""
        
        self.stats['total_frames'] += 1
        
        # 1. DETECCIÓN
        self.perf_monitor.start_timer('detection')
        detections = self.detector.detect(frame)
        self.perf_monitor.stop_timer('detection')
        
        self.stats['total_detections'] += len(detections)
        
        if not detections:
            return frame
        
        # 2. EMBEDDING EXTRACTION
        self.perf_monitor.start_timer('embedding')
        for det in detections:
            det.embedding = self.extractor.extract(frame, det)
            det.frame_idx = frame_idx
        self.perf_monitor.stop_timer('embedding')
        
        # 3. GNN ENHANCEMENT (si está disponible)
        if self.gnn and len(detections) > 1:
            self.perf_monitor.start_timer('gnn')
            try:
                graph_data = self.gnn.build_graph(detections)
                if graph_data is not None:
                    with torch.no_grad():
                        enhanced = self.gnn(
                            graph_data.x.to(self.config.device),
                            graph_data.edge_index.to(self.config.device),
                            graph_data.edge_attr.to(self.config.device) if graph_data.edge_attr is not None else None
                        )
                    
                    enhanced = enhanced.cpu().numpy()
                    
                    # Actualizar embeddings
                    valid_idx = 0
                    for det in detections:
                        if det.embedding is not None:
                            # Mezcla suave con embedding original
                            det.embedding = 0.7 * det.embedding + 0.3 * enhanced[valid_idx]
                            valid_idx += 1
            except Exception as e:
                if self.config.verbose > 1:
                    logger.debug(f"GNN error: {e}")
            self.perf_monitor.stop_timer('gnn')
        
        # 4. RECONOCIMIENTO CON BAYESIAN UNCERTAINTY
        self.perf_monitor.start_timer('recognition')
        current_threshold = self.threshold_controller.current_threshold
        
        for det in detections:
            if det.embedding is None or not self.gallery:
                continue
            
            if self.bayesian:
                # Reconocimiento bayesiano
                person_id, confidence, uncertainty = self.bayesian.compute_uncertainty(
                    det.embedding, self.gallery
                )
                
                det.recognition_uncertainty = uncertainty
                
                # Solo aceptar si uncertainty es baja
                if uncertainty < self.config.uncertainty_threshold and confidence > current_threshold:
                    det.person_id = person_id
                    det.recognition_confidence = confidence
                    self.stats['total_recognitions'] += 1
            else:
                # Reconocimiento estándar
                best_match = None
                best_sim = current_threshold
                
                for person_id, gallery_emb in self.gallery.items():
                    sim = 1 - cosine(det.embedding, gallery_emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_match = person_id
                
                if best_match:
                    det.person_id = best_match
                    det.recognition_confidence = best_sim
                    self.stats['total_recognitions'] += 1
        
        self.perf_monitor.stop_timer('recognition')
        
        # 5. TRACKING
        self.perf_monitor.start_timer('tracking')
        tracks = self.tracker.update(detections, frame_idx)
        self.perf_monitor.stop_timer('tracking')
        
        # 6. BEHAVIOR ANALYSIS
        if self.behavior:
            for track in tracks:
                if len(track.detections) >= self.config.min_behavior_samples:
                    positions = [d.center() for d in track.detections]
                    frame_indices = [d.frame_idx for d in track.detections]
                    
                    # Buscar tracks cercanos
                    nearby = []
                    if track.last_position():
                        for other_track in tracks:
                            if other_track.track_id == track.track_id:
                                continue
                            if other_track.last_position():
                                dist = np.linalg.norm(
                                    np.array(track.last_position()) - np.array(other_track.last_position())
                                )
                                if dist < self.config.social_proximity_threshold:
                                    nearby.append(other_track.track_id)
                    
                    behavior, behavior_conf = self.behavior.analyze(
                        track.track_id, positions, frame_indices, nearby
                    )
                    
                    track.behavior_pattern = behavior
                    track.behavior_history.append(behavior)
        
        # 7. UPDATE ADAPTIVE THRESHOLD
        quality_scores = [d.quality_score for d in detections]
        recognition_confs = [d.recognition_confidence for d in detections if d.person_id]
        new_threshold = self.threshold_controller.update(quality_scores, recognition_confs)
        self.stats['threshold_history'].append(new_threshold)
        
        # 8. VISUALIZACIÓN
        viz_frame = self.visualize(frame, detections, tracks, frame_idx)
        
        return viz_frame
    
    def visualize(self, frame: np.ndarray, detections: List[Detection], 
                 tracks: List[Track], frame_idx: int) -> np.ndarray:
        """Renderiza frame con anotaciones avanzadas"""
        viz = frame.copy()
        
        # Header con métricas
        current_threshold = self.threshold_controller.current_threshold
        active_tracks = len([t for t in tracks if t.disappeared < 5])
        
        header_text = f"NEXUS v8.0 ULTIMATE | Frame {frame_idx} | Detections: {len(detections)} | Tracks: {active_tracks} | Threshold: {current_threshold:.2f}"
        cv2.putText(viz, header_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Social graph
        if self.config.draw_social_graph and len(detections) > 1:
            for i, det_i in enumerate(detections):
                for j, det_j in enumerate(detections[i+1:], i+1):
                    c_i = det_i.center()
                    c_j = det_j.center()
                    dist = np.linalg.norm(np.array(c_i) - np.array(c_j))
                    
                    if dist < self.config.social_proximity_threshold:
                        alpha = 1.0 - (dist / self.config.social_proximity_threshold)
                        color_intensity = int(150 + 105 * alpha)
                        cv2.line(
                            viz,
                            (int(c_i[0]), int(c_i[1])),
                            (int(c_j[0]), int(c_j[1])),
                            (color_intensity, color_intensity, 255), 1, cv2.LINE_AA
                        )
        
        # Detections y tracks
        for det in detections:
            x1, y1, x2, y2 = det.bbox.astype(int)
            
            # Color según reconocimiento y calidad
            if det.person_id:
                conf = det.recognition_confidence
                if conf > 0.8:
                    color = (0, 255, 0)  # Verde brillante
                elif conf > 0.65:
                    color = (0, 200, 255)  # Amarillo
                else:
                    color = (0, 150, 255)  # Naranja
            else:
                # Color según calidad
                if det.quality_level == QualityLevel.EXCELLENT:
                    color = (255, 0, 255)  # Magenta
                elif det.quality_level == QualityLevel.GOOD:
                    color = (255, 255, 0)  # Cyan
                else:
                    color = (0, 0, 255)  # Rojo
            
            # BBox con grosor según calidad
            thickness = 2 if det.quality_level.value >= 3 else 1
            cv2.rectangle(viz, (x1, y1), (x2, y2), color, thickness)
            
            # Label con información completa
            label_parts = []
            
            if self.config.draw_track_ids and det.track_id:
                label_parts.append(f"T{det.track_id}")
            
            if det.person_id:
                label_parts.append(f"{det.person_id} {det.recognition_confidence*100:.0f}%")
                
                if self.config.draw_uncertainty and hasattr(det, 'recognition_uncertainty'):
                    label_parts.append(f"U:{det.recognition_uncertainty:.2f}")
            else:
                label_parts.append(f"Unknown Q:{det.quality_score:.2f}")
            
            label = " | ".join(label_parts)
            
            # Background para label
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(viz, (x1, y1-h-8), (x1+w+4, y1), color, -1)
            cv2.putText(viz, label, (x1+2, y1-4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Confidence bar
            if self.config.draw_confidence_bars and det.person_id:
                bar_w = int(100 * det.recognition_confidence)
                cv2.rectangle(viz, (x1, y2+5), (x1+bar_w, y2+15), color, -1)
                cv2.rectangle(viz, (x1, y2+5), (x1+100, y2+15), color, 1)
            
            # Motion vector
            if self.config.draw_motion_vectors and det.velocity:
                center = det.center()
                end_x = int(center[0] + det.velocity[0] * 10)
                end_y = int(center[1] + det.velocity[1] * 10)
                cv2.arrowedLine(viz, (int(center[0]), int(center[1])), (end_x, end_y),
                              (255, 0, 255), 2, tipLength=0.3)
        
        # Behavior labels
        if self.config.draw_behavior_labels:
            for track in tracks:
                if track.behavior_pattern and track.last_position():
                    pos = track.last_position()
                    behavior_text = track.behavior_pattern
                    
                    # Encontrar detección correspondiente
                    for det in detections:
                        if det.track_id == track.track_id:
                            x1, y1, x2, y2 = det.bbox.astype(int)
                            cv2.putText(viz, behavior_text, (x1, y2+30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
                            break
        
        return viz
    
    def export_results(self):
        """Exporta resultados y análisis"""
        logger.info("\n💾 Exportando resultados...")
        
        # JSON con estadísticas
        if self.config.save_json:
            avg_time = np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0
            avg_fps = 1.0 / avg_time if avg_time > 0 else 0
            
            perf_stats = self.perf_monitor.get_all_stats()
            
            results = {
                'metadata': {
                    'version': '8.0.0-ULTIMATE',
                    'timestamp': datetime.now().isoformat(),
                    'video': self.config.video_path,
                    'device': self.config.device,
                },
                'statistics': {
                    'total_frames': self.stats['total_frames'],
                    'total_detections': self.stats['total_detections'],
                    'total_recognitions': self.stats['total_recognitions'],
                    'avg_fps': float(avg_fps),
                    'recognition_rate': self.stats['total_recognitions'] / max(1, self.stats['total_detections']),
                    'reid_successes': self.stats.get('reid_successes', 0),
                },
                'performance': {
                    name: {
                        'mean_ms': stats['mean'] * 1000,
                        'std_ms': stats['std'] * 1000,
                        'count': stats['count']
                    }
                    for name, stats in perf_stats.items()
                },
                'features': {
                    'dynamic_fusion': self.config.enable_dynamic_fusion,
                    'gnn': self.gnn is not None,
                    'bayesian_uncertainty': self.bayesian is not None,
                    'physics_motion': self.physics is not None,
                    'behavior_analysis': self.behavior is not None,
                    'active_reid': self.config.enable_active_reid,
                    'adaptive_threshold': True,
                },
                'gallery': {
                    'persons': len(self.gallery),
                    'names': list(self.gallery.keys()),
                },
                'tracking': {
                    'total_tracks': self.tracker.next_id - 1,
                    'active_tracks': len(self.tracker.get_active_tracks()),
                },
                'adaptive_threshold': {
                    'initial': self.config.base_threshold,
                    'final': self.threshold_controller.current_threshold,
                    'history': self.stats['threshold_history'][-100:] if self.stats['threshold_history'] else []
                }
            }
            
            with open(f'{self.config.output_dir}/nexus_ultimate_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info("   ✅ JSON guardado")
        
        # Embeddings
        if self.config.save_embeddings and self.gallery:
            with open(f'{self.config.output_dir}/gallery_embeddings.pkl', 'wb') as f:
                pickle.dump(self.gallery, f)
            logger.info("   ✅ Embeddings guardados")
        
        # Visualizaciones
        if self.config.save_heatmaps and self.stats['threshold_history']:
            plt.figure(figsize=(12, 6))
            plt.plot(self.stats['threshold_history'], linewidth=2)
            plt.axhline(y=self.config.base_threshold, color='r', linestyle='--', 
                       label='Threshold inicial')
            plt.xlabel('Frame')
            plt.ylabel('Recognition Threshold')
            plt.title('Adaptive Threshold Evolution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{self.config.output_dir}/threshold_evolution.png', dpi=150)
            plt.close()
            logger.info("   ✅ Visualizaciones guardadas")
    
    def print_summary(self):
        """Imprime resumen final"""
        avg_time = np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        print("\n" + "="*80)
        print("🌟 NEXUS v8.0 ULTIMATE - RESUMEN DEL PROCESAMIENTO")
        print("="*80)
        print(f"⏱️  Frames procesados: {self.stats['total_frames']}")
        print(f"👤 Total detecciones: {self.stats['total_detections']}")
        print(f"✅ Total reconocimientos: {self.stats['total_recognitions']}")
        print(f"📈 Tasa reconocimiento: {self.stats['total_recognitions']/max(1, self.stats['total_detections'])*100:.1f}%")
        print(f"⚡ FPS promedio: {avg_fps:.1f}")
        print(f"🎯 Threshold final: {self.threshold_controller.current_threshold:.3f}")
        
        # Performance breakdown
        perf_stats = self.perf_monitor.get_all_stats()
        if perf_stats:
            print("\n⏰ DESGLOSE DE TIEMPOS:")
            for name, stats in sorted(perf_stats.items(), key=lambda x: x[1]['mean'], reverse=True):
                print(f"   {name:20s}: {stats['mean']*1000:6.2f} ms ± {stats['std']*1000:5.2f} ms")
        
        print("="*80)
        print(f"\n📁 Resultados en: {self.config.output_dir}/")
        print("🎬 Video: output.mp4")
        print("📊 JSON: nexus_ultimate_results.json")
        
        if self.config.save_heatmaps:
            print("📈 Gráficas: threshold_evolution.png")
        
        print("\n" + "="*80)
        print("🌟 NEXUS v8.0 ULTIMATE - El futuro del reconocimiento facial")
        print("="*80 + "\n")

# ===================================================================
# MAIN ENTRY POINT
# ===================================================================
def main():
    """Punto de entrada principal"""
    
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║               NEXUS v8.0 ULTIMATE                                          ║
║                                                                            ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Crear configuración
        config = UltimateConfig()
        
        # Validar que existen archivos necesarios
        if not Path(config.video_path).exists():
            logger.error(f"❌ Video no encontrado: {config.video_path}")
            sys.exit(1)
        
        # Crear y ejecutar sistema
        system = NexusUltimateSystem(config)
        system.process_video()
        
        print("\n" + "="*80)
        print("🎉 PROCESAMIENTO COMPLETADO CON ÉXITO")
        print("="*80)
        print(f"\n📁 Todos los resultados en: {config.output_dir}/")
        print("\n🌟 NEXUS v8.0 ULTIMATE - El mejor sistema del mercado")
        print("="*80 + "\n")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Procesamiento interrumpido por usuario")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
