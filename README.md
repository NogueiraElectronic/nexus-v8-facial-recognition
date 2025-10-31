face_quality:
  min_size: 80
  max_size: 2000
  min_sharpness: 50
  min_brightness: 40
  max_brightness: 220
  min_contrast: 20
  max_head_pose:
    yaw: 30
    pitch: 30
    roll: 15
  min_eye_distance: 40
  require_both_eyes_visible: true
  require_mouth_visible: true
  max_occlusion_percent: 20
  allow_glasses: true
  allow_mask: false
  
image_quality:
  supported_formats: [jpg, jpeg, png, bmp]
  max_file_size_mb: 10
  min_resolution: [640, 480]
  max_resolution: [4096, 4096]
  compression_quality: 95

enrollment_requirements:
  min_quality_score: 70
  min_samples: 3
  max_samples: 10
  sample_diversity_threshold: 0.3
  require_liveness_check: true
