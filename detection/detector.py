"""
YOLO-based entity detection for AoE2.

Provides fast, accurate detection of game entities (units, buildings, resources)
with bounding boxes and semantic IDs for action targeting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING
import io

if TYPE_CHECKING:
    from PIL import Image


@dataclass
class DetectedEntity:
    """Represents a detected game entity."""
    id: str                          # Unique ID: e.g., "sheep_0", "villager_1"
    class_name: str                  # Entity class: "sheep", "villager", "tc"
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    center: tuple[float, float]      # Center point (x, y)
    confidence: float                # Detection confidence 0-1
    area: float = field(default=0)   # Bounding box area in pixels

    def to_dict(self) -> dict:
        """Convert to dictionary for LLM context."""
        return {
            "id": self.id,
            "class": self.class_name,
            "bbox": list(self.bbox),
            "center": self.center,
            "confidence": self.confidence
        }


# Default class names for AoE2 detection
DEFAULT_CLASSES = [
    "sheep",           # 0 - Food source
    "villager",        # 1 - Worker unit
    "town_center",     # 2 - Main building
    "house",           # 3 - Population building
    "barracks",        # 4 - Military building
    "mill",            # 5 - Food drop-off
    "lumber_camp",     # 6 - Wood drop-off
    "mining_camp",     # 7 - Gold/stone drop-off
    "scout",           # 8 - Scout cavalry
    "deer",            # 9 - Food source
    "boar",            # 10 - Food source (dangerous)
    "berries",         # 11 - Food source
    "gold_mine",       # 12 - Gold resource
    "stone_mine",      # 13 - Stone resource
    "tree",            # 14 - Wood resource
    "farm",            # 15 - Food building
    "archer",          # 16 - Military unit
    "spearman",        # 17 - Military unit
    "enemy_unit",      # 18 - Any enemy unit
    "enemy_building",  # 19 - Any enemy building
]


class EntityDetector:
    """YOLO-based entity detector for AoE2 screenshots.

    Supports both real YOLO models and a mock mode for development/testing.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        class_names: Optional[list[str]] = None,
        confidence_threshold: float = 0.5,
        use_mock: bool = False
    ):
        """Initialize the detector.

        Args:
            model_path: Path to YOLO .pt weights file
            class_names: List of class names (order matches model output)
            confidence_threshold: Minimum confidence for detections
            use_mock: If True, use mock detections (for testing without model)
        """
        self.class_names = class_names or DEFAULT_CLASSES
        self.confidence_threshold = confidence_threshold
        self.use_mock = use_mock
        self.model = None
        self._class_counters: dict[str, int] = {}

        if model_path and not use_mock:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load YOLO model from weights file."""
        try:
            from ultralytics import YOLO
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            self.model = YOLO(str(path))
            self.use_mock = False
        except ImportError:
            print("WARNING: ultralytics not installed. Using mock detection.")
            self.use_mock = True
        except Exception as e:
            print(f"WARNING: Failed to load YOLO model: {e}. Using mock detection.")
            self.use_mock = True

    def _reset_counters(self):
        """Reset entity ID counters for new detection."""
        self._class_counters = {name: 0 for name in self.class_names}

    def _generate_id(self, class_name: str) -> str:
        """Generate unique ID for detected entity."""
        if class_name not in self._class_counters:
            self._class_counters[class_name] = 0
        idx = self._class_counters[class_name]
        self._class_counters[class_name] += 1
        return f"{class_name}_{idx}"

    def detect(self, screenshot: Union[bytes, "Image.Image"]) -> list[DetectedEntity]:
        """Detect entities in screenshot.

        Args:
            screenshot: JPEG bytes or PIL Image

        Returns:
            List of detected entities sorted by class, then by confidence
        """
        self._reset_counters()

        if self.use_mock:
            return self._mock_detect(screenshot)

        return self._yolo_detect(screenshot)

    def _yolo_detect(self, screenshot: Union[bytes, "Image.Image"]) -> list[DetectedEntity]:
        """Run actual YOLO detection."""
        from PIL import Image

        # Convert bytes to PIL Image if needed
        if isinstance(screenshot, bytes):
            image = Image.open(io.BytesIO(screenshot))
        else:
            image = screenshot

        # Run inference
        results = self.model(image, conf=self.confidence_threshold, verbose=False)

        entities = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i, (box, cls_id, conf) in enumerate(zip(
                boxes.xyxy.cpu().numpy(),
                boxes.cls.cpu().numpy(),
                boxes.conf.cpu().numpy()
            )):
                x1, y1, x2, y2 = box.tolist()
                class_idx = int(cls_id)

                # Get class name (with fallback)
                if class_idx < len(self.class_names):
                    class_name = self.class_names[class_idx]
                else:
                    class_name = f"unknown_{class_idx}"

                # Calculate center and area
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                area = (x2 - x1) * (y2 - y1)

                entity = DetectedEntity(
                    id=self._generate_id(class_name),
                    class_name=class_name,
                    bbox=(x1, y1, x2, y2),
                    center=(center_x, center_y),
                    confidence=float(conf),
                    area=area
                )
                entities.append(entity)

        # Sort by class name, then by confidence (highest first)
        entities.sort(key=lambda e: (e.class_name, -e.confidence))

        return entities

    def _mock_detect(self, screenshot: Union[bytes, "Image.Image"]) -> list[DetectedEntity]:
        """Generate mock detections for testing.

        Creates plausible detections based on typical Dark Age game state.
        """
        from PIL import Image
        import random

        # Get image dimensions
        if isinstance(screenshot, bytes):
            image = Image.open(io.BytesIO(screenshot))
            width, height = image.size
        elif hasattr(screenshot, 'size'):
            width, height = screenshot.size
        else:
            width, height = 1920, 1080  # Default fallback

        # Set random seed for reproducibility within same session
        random.seed(42)

        entities = []

        # Town Center - usually center of screen
        tc_x = width * 0.5 + random.uniform(-100, 100)
        tc_y = height * 0.5 + random.uniform(-50, 50)
        entities.append(DetectedEntity(
            id=self._generate_id("town_center"),
            class_name="town_center",
            bbox=(tc_x - 80, tc_y - 60, tc_x + 80, tc_y + 60),
            center=(tc_x, tc_y),
            confidence=0.95,
            area=160 * 120
        ))

        # Sheep - typically 2-4 near TC at game start
        for i in range(random.randint(2, 4)):
            sheep_x = tc_x + random.uniform(-200, 200)
            sheep_y = tc_y + random.uniform(-150, 150)
            entities.append(DetectedEntity(
                id=self._generate_id("sheep"),
                class_name="sheep",
                bbox=(sheep_x - 15, sheep_y - 10, sheep_x + 15, sheep_y + 10),
                center=(sheep_x, sheep_y),
                confidence=random.uniform(0.7, 0.95),
                area=30 * 20
            ))

        # Villagers - 3 starting villagers
        for i in range(3):
            vill_x = tc_x + random.uniform(-150, 150)
            vill_y = tc_y + random.uniform(-100, 100)
            entities.append(DetectedEntity(
                id=self._generate_id("villager"),
                class_name="villager",
                bbox=(vill_x - 12, vill_y - 20, vill_x + 12, vill_y + 5),
                center=(vill_x, vill_y),
                confidence=random.uniform(0.75, 0.92),
                area=24 * 25
            ))

        # Scout - usually exploring
        scout_x = random.uniform(100, width - 100)
        scout_y = random.uniform(100, height - 100)
        entities.append(DetectedEntity(
            id=self._generate_id("scout"),
            class_name="scout",
            bbox=(scout_x - 15, scout_y - 18, scout_x + 15, scout_y + 8),
            center=(scout_x, scout_y),
            confidence=0.88,
            area=30 * 26
        ))

        # Sort by class name, then by confidence
        entities.sort(key=lambda e: (e.class_name, -e.confidence))

        return entities

    def detect_to_dict_list(self, screenshot: Union[bytes, "Image.Image"]) -> list[dict]:
        """Detect and return as list of dictionaries.

        Convenience method for LLM context building.
        """
        return [e.to_dict() for e in self.detect(screenshot)]

    def find_entity_by_id(
        self,
        entities: list[DetectedEntity],
        target_id: str
    ) -> Optional[DetectedEntity]:
        """Find an entity by its ID.

        Args:
            entities: List of detected entities
            target_id: Entity ID to find (e.g., "sheep_0")

        Returns:
            The matching entity, or None if not found
        """
        for entity in entities:
            if entity.id == target_id:
                return entity
        return None

    def find_entities_by_class(
        self,
        entities: list[DetectedEntity],
        class_name: str
    ) -> list[DetectedEntity]:
        """Find all entities of a given class.

        Args:
            entities: List of detected entities
            class_name: Class name to filter by

        Returns:
            List of matching entities (sorted by confidence)
        """
        matches = [e for e in entities if e.class_name == class_name]
        matches.sort(key=lambda e: -e.confidence)
        return matches

    def find_nearest_entity(
        self,
        entities: list[DetectedEntity],
        point: tuple[float, float],
        class_filter: Optional[str] = None
    ) -> Optional[DetectedEntity]:
        """Find the nearest entity to a point.

        Args:
            entities: List of detected entities
            point: (x, y) coordinates
            class_filter: Optional class name to filter by

        Returns:
            Nearest entity, or None if no entities
        """
        candidates = entities
        if class_filter:
            candidates = self.find_entities_by_class(entities, class_filter)

        if not candidates:
            return None

        def distance(e: DetectedEntity) -> float:
            dx = e.center[0] - point[0]
            dy = e.center[1] - point[1]
            return (dx * dx + dy * dy) ** 0.5

        return min(candidates, key=distance)


# Singleton instance for easy access
_instance: Optional[EntityDetector] = None

def get_detector(model_path: Optional[str] = None, use_mock: bool = False) -> EntityDetector:
    """Get or create the singleton detector instance."""
    global _instance
    if _instance is None:
        if model_path is None:
            # Default model path
            model_path = str(Path(__file__).parent / "models" / "aoe2_yolo.pt")
        _instance = EntityDetector(model_path=model_path, use_mock=use_mock)
    return _instance
