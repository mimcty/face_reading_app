import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, Union, Tuple, List
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans


class FaceDetector:
    def __init__(self):
        # Initialize face mesh with high accuracy settings
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )

        # Comprehensive landmark indices for Chinese face reading
        self.LANDMARK_INDICES = {
            # Core facial structure
            "jawline": [*range(234, 454, 10)],  # Sparse jawline points
            "forehead": [10, 67, 69, 104, 108, 109, 151, 299, 337],
            "cheekbones": [116, 117, 118, 119, 120, 121, 346, 347, 348, 349, 350, 351],

            # Detailed facial features
            "left_eye": [33, 133, 159, 145, 158, 153, 144, 154],
            "right_eye": [263, 362, 386, 374, 385, 380, 373, 381],
            "left_eyebrow": [46, 53, 52, 65, 55, 107, 66, 105],
            "right_eyebrow": [276, 283, 282, 295, 285, 336, 296, 334],
            "nose": [1, 2, 3, 4, 5, 6, 98, 168, 327, 358, 412],
            "mouth": [61, 185, 40, 39, 37, 0, 267, 269, 270, 291, 375, 321, 405],

            # Additional reference points
            "philtrum": [2, 195, 197, 5],
            "chin": [152, 148, 176, 149, 150, 136, 172, 58, 132],
            "temples": [162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152]
        }

    def get_measurements(self, image_data: Union[str, bytes]) -> Dict:
        """Main method to analyze a face and return comprehensive measurements"""
        try:
            img = self._load_image(image_data)
            if img is None:
                return {"error": "Invalid image data"}

            # Preprocess image for better detection
            img = self._preprocess_image(img)

            # Get validated facial landmarks
            landmarks = self._get_validated_landmarks(img)
            if isinstance(landmarks, dict) and "error" in landmarks:
                return landmarks

            # Calculate geometric measurements
            measurements = self._calculate_measurements(img.shape, landmarks)

            # Analyze facial features
            traits = {
                "eyes": self._analyze_eyes(landmarks),
                "eyebrows": self._analyze_eyebrows(landmarks),
                "nose": self._analyze_nose(landmarks),
                "mouth": self._analyze_mouth(landmarks),
                "jawline": self._analyze_jawline(landmarks),
                "symmetry": self._analyze_symmetry(landmarks)
            }

            # Analyze skin tone and complexion
            skin = self._analyze_skin_tone(img, landmarks)

            # Classify face shape and element
            shape = self._classify_face_shape(measurements)
            element, breakdown, scores = self._infer_element(traits, shape, skin)

            return {
                "measurements": measurements,
                "traits": traits,
                "skin_tone": skin["tone"],
                "qi": skin["qi"],
                "shape": shape,
                "element": element,
                "confidence": {"shape_conf": 0.9},
                "element_breakdown": breakdown,
                "element_scores": scores
            }

        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

    # ========== Core Analysis Methods ==========

    def _analyze_eyes(self, points: Dict[str, np.ndarray]) -> Dict:
        """Detailed eye shape analysis for Chinese face reading"""
        left_eye = points["left_eye"]
        right_eye = points["right_eye"]

        # Calculate eye aspect ratios
        left_ear = self._eye_aspect_ratio(left_eye)
        right_ear = self._eye_aspect_ratio(right_eye)

        # Determine eye shapes
        left_shape = self._classify_eye_shape(left_ear)
        right_shape = self._classify_eye_shape(right_ear)

        # Calculate eye angles
        left_angle = self._eye_tilt_angle(left_eye)
        right_angle = self._eye_tilt_angle(right_eye)

        return {
            "left": {
                "aspect_ratio": left_ear,
                "shape": left_shape,
                "tilt_angle": left_angle,
                "size": self._calculate_eye_size(left_eye)
            },
            "right": {
                "aspect_ratio": right_ear,
                "shape": right_shape,
                "tilt_angle": right_angle,
                "size": self._calculate_eye_size(right_eye)
            },
            "asymmetry": abs(left_ear - right_ear)
        }

    def _analyze_eyebrows(self, points: Dict[str, np.ndarray]) -> Dict:
        """Eyebrow shape and curvature analysis"""
        left_brow = points["left_eyebrow"]
        right_brow = points["right_eyebrow"]

        # Calculate curvature
        left_curve = self._calculate_eyebrow_curvature(left_brow)
        right_curve = self._calculate_eyebrow_curvature(right_brow)

        # Determine eyebrow shapes
        left_shape = self._classify_eyebrow_shape(left_curve)
        right_shape = self._classify_eyebrow_shape(right_curve)

        return {
            "left": {
                "curvature": left_curve,
                "shape": left_shape,
                "thickness": self._calculate_eyebrow_thickness(left_brow)
            },
            "right": {
                "curvature": right_curve,
                "shape": right_shape,
                "thickness": self._calculate_eyebrow_thickness(right_brow)
            },
            "asymmetry": abs(left_curve - right_curve)
        }

    def _analyze_nose(self, points: Dict[str, np.ndarray]) -> Dict:
        """Detailed nose shape analysis"""
        nose_points = points["nose"]

        # Calculate nose dimensions
        length = np.linalg.norm(nose_points[0] - nose_points[-1])
        width = np.linalg.norm(nose_points[3] - nose_points[7])

        # Analyze nose shape
        shape = self._classify_nose_shape(length, width)

        return {
            "length": length,
            "width": width,
            "ratio": width / length,
            "shape": shape,
            "bridge": self._analyze_nose_bridge(nose_points)
        }

    def _analyze_mouth(self, points: Dict[str, np.ndarray]) -> Dict:
        """Mouth shape and proportion analysis"""
        mouth_points = points["mouth"]

        # Calculate mouth dimensions
        width = np.linalg.norm(mouth_points[0] - mouth_points[6])
        height = np.linalg.norm(mouth_points[3] - mouth_points[9])

        # Analyze lip shape
        shape = self._classify_mouth_shape(width, height)

        return {
            "width": width,
            "height": height,
            "ratio": height / width,
            "shape": shape,
            "corners": self._analyze_mouth_corners(mouth_points)
        }

    def _analyze_jawline(self, points: Dict[str, np.ndarray]) -> Dict:
        """Jawline shape and angle analysis"""
        jaw_points = points["jawline"]

        # Calculate jaw angle
        angle = self._calculate_jaw_angle(jaw_points)

        # Analyze jaw shape
        shape = self._classify_jawline_shape(jaw_points)

        return {
            "angle": angle,
            "shape": shape,
            "sharpness": self._calculate_jaw_sharpness(jaw_points)
        }

    def _analyze_symmetry(self, points: Dict[str, np.ndarray]) -> Dict:
        """Comprehensive facial symmetry analysis"""
        # Eye symmetry
        left_eye = points["left_eye"]
        right_eye = points["right_eye"]
        eye_sym = 1 - (np.linalg.norm(left_eye[0] - right_eye[0]) /
                       np.linalg.norm(points["forehead"][0] - points["chin"][0]))

        # Eyebrow symmetry
        left_brow = points["left_eyebrow"]
        right_brow = points["right_eyebrow"]
        brow_sym = 1 - (np.linalg.norm(left_brow[0] - right_brow[0]) /
                        np.linalg.norm(points["forehead"][0] - points["chin"][0]))

        # Jaw symmetry
        jaw_sym = self._calculate_jaw_symmetry(points["jawline"])

        return {
            "eye_symmetry": eye_sym,
            "eyebrow_symmetry": brow_sym,
            "jaw_symmetry": jaw_sym,
            "overall_symmetry": (eye_sym + brow_sym + jaw_sym) / 3
        }

    def _analyze_skin_tone(self, img: np.ndarray, points: Dict[str, np.ndarray]) -> Dict:
        """Advanced skin tone and complexion analysis"""
        # Sample skin regions
        regions = {
            "forehead": self._get_skin_region(img, points["forehead"][0]),
            "left_cheek": self._get_skin_region(img, points["cheekbones"][0]),
            "right_cheek": self._get_skin_region(img, points["cheekbones"][-1]),
            "chin": self._get_skin_region(img, points["chin"][0])
        }

        # Convert to LAB color space
        lab_values = {}
        for name, region in regions.items():
            lab = cv2.cvtColor(region, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            lab_values[name] = {
                "lightness": np.mean(l),
                "warmth": np.mean(a),
                "coolness": np.mean(b)
            }

        # Calculate overall skin tone
        avg_lightness = np.mean([v["lightness"] for v in lab_values.values()])
        avg_warmth = np.mean([v["warmth"] for v in lab_values.values()])
        avg_coolness = np.mean([v["coolness"] for v in lab_values.values()])

        # Traditional Chinese Medicine diagnosis
        complexion = self._diagnose_complexion(avg_warmth, avg_coolness)

        return {
            "tone": {
                "lightness": avg_lightness,
                "warmth": avg_warmth,
                "coolness": avg_coolness
            },
            "qi": {
                "level": "strong" if avg_lightness > 140 else "weak",
                "complexion": complexion,
                "balance": self._calculate_qi_balance(lab_values)
            }
        }

    # ========== Classification Methods ==========

    def _classify_face_shape(self, measurements: Dict) -> str:
        """Classify face shape based on measurements"""
        ratio = measurements["face_width"] / measurements["face_height"]
        jaw_angle = measurements["jaw_angle"]

        if ratio > 0.95 and jaw_angle > 135:
            return "round"
        elif ratio < 0.72:
            return "long"
        elif 0.75 <= ratio <= 0.85 and jaw_angle > 130:
            return "oval"
        elif jaw_angle < 115 and ratio < 0.8:
            return "heart"
        elif ratio > 0.85 and jaw_angle > 120:
            return "triangle"
        else:
            return "square"

    def _classify_eye_shape(self, ear: float) -> str:
        """Classify eye shape based on aspect ratio"""
        if ear < 0.25:
            return "willow_leaf"
        elif ear < 0.35:
            return "phoenix"
        elif ear < 0.45:
            return "dragon"
        else:
            return "peach_blossom"

    def _classify_eyebrow_shape(self, curvature: float) -> str:
        """Classify eyebrow shape based on curvature"""
        if curvature > 160:
            return "straight"
        elif curvature > 135:
            return "arched"
        else:
            return "angled"

    def _classify_nose_shape(self, length: float, width: float) -> str:
        """Classify nose shape based on proportions"""
        ratio = width / length
        if ratio > 0.8:
            return "broad"
        elif ratio < 0.5:
            return "narrow"
        else:
            return "classic"

    def _classify_mouth_shape(self, width: float, height: float) -> str:
        """Classify mouth shape based on proportions"""
        ratio = height / width
        if ratio > 0.3:
            return "full"
        elif ratio < 0.15:
            return "thin"
        else:
            return "medium"

    def _classify_jawline_shape(self, jaw_points: np.ndarray) -> str:
        """Classify jawline shape based on points"""
        angles = []
        for i in range(1, len(jaw_points) - 1):
            angle = self._angle_between(jaw_points[i - 1], jaw_points[i], jaw_points[i + 1])
            angles.append(angle)

        avg_angle = np.mean(angles)
        if avg_angle > 150:
            return "square"
        elif avg_angle > 130:
            return "rounded"
        else:
            return "pointed"

    def _infer_element(self, traits: Dict, shape: str, skin: Dict) -> Tuple[str, Dict, Dict]:
        """Infer Chinese five elements based on facial features"""
        element_scores = {
            "wood": 0,
            "fire": 0,
            "earth": 0,
            "metal": 0,
            "water": 0
        }
        breakdown = {}

        # Face shape contribution
        shape_element = {
            "round": "metal",
            "long": "wood",
            "oval": "wood",
            "heart": "fire",
            "triangle": "water",
            "square": "earth"
        }.get(shape, "earth")
        element_scores[shape_element] += 0.3
        breakdown["face_shape"] = (shape_element, f"{shape} face")

        # Eye shape contribution
        eye_element = {
            "phoenix": "fire",
            "willow_leaf": "wood",
            "dragon": "fire",
            "peach_blossom": "water"
        }.get(traits["eyes"]["left"]["shape"], "water")
        element_scores[eye_element] += 0.2
        breakdown["eyes"] = (eye_element, f"{traits['eyes']['left']['shape']} eyes")

        # Eyebrow contribution
        brow_element = {
            "straight": "metal",
            "arched": "wood",
            "angled": "water"
        }.get(traits["eyebrows"]["left"]["shape"], "wood")
        element_scores[brow_element] += 0.15
        breakdown["eyebrows"] = (brow_element, f"{traits['eyebrows']['left']['shape']} brows")

        # Nose contribution
        nose_element = "earth" if traits["nose"]["ratio"] > 0.7 else "metal"
        element_scores[nose_element] += 0.15
        breakdown["nose"] = (nose_element, f"nose ratio {traits['nose']['ratio']:.2f}")

        # Mouth contribution
        mouth_element = "fire" if traits["mouth"]["ratio"] > 0.25 else "water"
        element_scores[mouth_element] += 0.1
        breakdown["mouth"] = (mouth_element, f"mouth ratio {traits['mouth']['ratio']:.2f}")

        # Skin tone contribution
        skin_element = self._classify_skin_element(skin["tone"])
        element_scores[skin_element] += 0.1
        breakdown["skin"] = (skin_element, f"skin tone L:{skin['tone']['lightness']:.1f}")

        dominant = max(element_scores, key=element_scores.get)
        return dominant, breakdown, element_scores

    # ========== Helper Methods ==========

    def _load_image(self, image_data: Union[str, bytes]) -> np.ndarray:
        """Load image from path or bytes"""
        if isinstance(image_data, str):
            img = cv2.imread(image_data)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
        else:
            arr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Enhance image for better face detection"""
        # Convert to YCrCb and equalize luminance
        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

        # Mild sharpening
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)

    def _get_validated_landmarks(self, img: np.ndarray) -> Union[Dict, np.ndarray]:
        """Detect and validate facial landmarks"""
        results = self.face_mesh.process(img)
        if not results.multi_face_landmarks:
            return {"error": "No face detected"}

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = img.shape[:2]

        # Convert to pixel coordinates
        points = {}
        for region, indices in self.LANDMARK_INDICES.items():
            points[region] = np.array([(landmarks[i].x * w, landmarks[i].y * h)
                                       for i in indices])

        # Validate face proportions
        face_width = np.linalg.norm(points["jawline"][0] - points["jawline"][-1])
        face_height = np.linalg.norm(points["forehead"][0] - points["chin"][0])
        if face_width / face_height < 0.5 or face_width / face_height > 1.5:
            return {"error": "Invalid face proportions"}

        return points

    def _calculate_measurements(self, img_shape: Tuple, points: Dict) -> Dict:
        """Calculate geometric measurements of facial features"""
        h, w = img_shape[:2]

        return {
            "face_width": np.linalg.norm(points["jawline"][0] - points["jawline"][-1]),
            "face_height": np.linalg.norm(points["forehead"][0] - points["chin"][0]),
            "face_ratio": np.linalg.norm(points["jawline"][0] - points["jawline"][-1]) /
                          np.linalg.norm(points["forehead"][0] - points["chin"][0]),
            "jaw_angle": self._calculate_jaw_angle(points["jawline"]),
            "forehead_width": np.linalg.norm(points["temples"][0] - points["temples"][-1]),
            "cheekbone_width": np.linalg.norm(points["cheekbones"][0] - points["cheekbones"][-1]),
            "upper_third": np.linalg.norm(points["forehead"][0] - points["left_eyebrow"][0]),
            "middle_third": np.linalg.norm(points["left_eyebrow"][0] - points["nose"][0]),
            "lower_third": np.linalg.norm(points["nose"][0] - points["chin"][0])
        }

    def _eye_aspect_ratio(self, eye_points: np.ndarray) -> float:
        """Calculate eye aspect ratio (EAR)"""
        # Vertical distances
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[6])

        # Horizontal distance
        h = np.linalg.norm(eye_points[0] - eye_points[3])

        return (v1 + v2) / (2.0 * h)

    def _eye_tilt_angle(self, eye_points: np.ndarray) -> float:
        """Calculate eye tilt angle in degrees"""
        return np.degrees(np.arctan2(
            eye_points[3][1] - eye_points[0][1],
            eye_points[3][0] - eye_points[0][0]
        ))

    def _calculate_eye_size(self, eye_points: np.ndarray) -> float:
        """Calculate relative eye size"""
        area = 0.5 * np.abs(
            np.dot(eye_points[:, 0], np.roll(eye_points[:, 1], 1)) -
            np.dot(eye_points[:, 1], np.roll(eye_points[:, 0], 1))
        )
        return area

    def _calculate_eyebrow_curvature(self, brow_points: np.ndarray) -> float:
        """Calculate eyebrow curvature in degrees"""
        return self._angle_between(brow_points[0], brow_points[len(brow_points) // 2], brow_points[-1])

    def _calculate_eyebrow_thickness(self, brow_points: np.ndarray) -> float:
        """Calculate average eyebrow thickness"""
        return np.mean([np.linalg.norm(brow_points[i] - brow_points[i + 1])
                        for i in range(len(brow_points) - 1)])

    def _analyze_nose_bridge(self, nose_points: np.ndarray) -> Dict:
        """Analyze nose bridge characteristics"""
        bridge_points = nose_points[:7]  # Use first 7 points for bridge
        angles = []
        for i in range(1, len(bridge_points) - 1):
            angles.append(self._angle_between(bridge_points[i - 1], bridge_points[i], bridge_points[i + 1]))

        return {
            "straightness": np.mean(angles),
            "width_variation": np.std([np.linalg.norm(bridge_points[i] - bridge_points[i + 1])
                                       for i in range(len(bridge_points) - 1)])
        }

    def _analyze_mouth_corners(self, mouth_points: np.ndarray) -> Dict:
        """Analyze mouth corner characteristics"""
        left_corner = mouth_points[0]
        right_corner = mouth_points[6]
        center_top = mouth_points[3]
        center_bottom = mouth_points[9]

        return {
            "angle": self._angle_between(left_corner, center_top, right_corner),
            "droop": (left_corner[1] + right_corner[1]) / 2 - (center_top[1] + center_bottom[1]) / 2
        }

    def _calculate_jaw_angle(self, jaw_points: np.ndarray) -> float:
        """Calculate jaw angle at chin point"""
        return self._angle_between(jaw_points[0], jaw_points[len(jaw_points) // 2], jaw_points[-1])

    def _calculate_jaw_sharpness(self, jaw_points: np.ndarray) -> float:
        """Calculate jawline sharpness metric"""
        angles = []
        for i in range(1, len(jaw_points) - 1):
            angles.append(self._angle_between(jaw_points[i - 1], jaw_points[i], jaw_points[i + 1]))
        return np.std(angles)  # Higher std = more variation = sharper angles

    def _calculate_jaw_symmetry(self, jaw_points: np.ndarray) -> float:
        """Calculate jaw symmetry metric (0-1)"""
        mid = len(jaw_points) // 2
        left = jaw_points[:mid]
        right = jaw_points[mid + 1:][::-1]  # Reverse to match left side

        if len(left) != len(right):
            return 0.5  # Can't compare directly

        diffs = [np.linalg.norm(l - r) for l, r in zip(left, right)]
        avg_diff = np.mean(diffs)
        max_diff = np.linalg.norm(jaw_points[0] - jaw_points[-1])

        return 1 - (avg_diff / max_diff)

    def _get_skin_region(self, img: np.ndarray, center: np.ndarray, size: int = 20) -> np.ndarray:
        """Extract square skin region from face"""
        x, y = int(center[0]), int(center[1])
        return img[max(0, y - size):min(img.shape[0], y + size),
               max(0, x - size):min(img.shape[1], x + size)]

    def _classify_skin_element(self, tone: Dict) -> str:
        """Classify skin tone into five elements"""
        if tone["warmth"] > 150:
            return "fire"
        elif tone["coolness"] > 140:
            return "water"
        elif tone["lightness"] < 50:
            return "metal"
        elif tone["lightness"] > 180:
            return "wood"
        else:
            return "earth"

    def _diagnose_complexion(self, warmth: float, coolness: float) -> str:
        """Traditional Chinese Medicine complexion diagnosis"""
        if warmth > 160:
            return "reddish (心火旺)"
        elif coolness > 150:
            return "yellowish (脾虚)"
        elif warmth < 120 and coolness < 120:
            return "pale (气血不足)"
        else:
            return "balanced"

    def _calculate_qi_balance(self, lab_values: Dict) -> Dict:
        """Calculate TCM qi balance across facial regions"""
        regions = list(lab_values.keys())
        a_values = [lab_values[r]["warmth"] for r in regions]
        b_values = [lab_values[r]["coolness"] for r in regions]

        return {
            "wood": abs(a_values[0] - a_values[2]),  # Forehead vs right cheek
            "fire": abs(a_values[1] - a_values[3]),  # Left cheek vs chin
            "earth": np.std(a_values),
            "metal": np.std(b_values),
            "water": abs(np.mean(a_values) - np.mean(b_values))
        }

    def _angle_between(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Calculate angle between three points in degrees"""
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))