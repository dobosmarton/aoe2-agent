# YOLO and object detection

An object detector takes an image and returns a list of boxes, each with a class from a fixed list and a confidence score. The project's detector is a fine-tuned Ultralytics YOLO26n (the `n` is the smallest size in the family). The concepts below are the ones the [entity detector](../detection/detector.md) and [serving and evaluation](../detection/serving-and-evaluation.md) pages assume.

## One forward pass, many predictions

YOLO is a single-stage detector: one pass through a convolutional network produces every candidate box at once. The network has three parts:

| Part | Does |
|---|---|
| Backbone | Turns pixels into feature maps at decreasing resolution |
| Neck | Combines coarse and fine feature maps so each scale sees both detail and context |
| Head | At every cell of each output grid, predicts a box and one score per class |

The grids have strides of 8, 16 and 32 input pixels. Fine grids catch small objects such as sheep, coarse grids catch town centres. Each grid cell is one prediction, which ONNX outputs and the project's code call an anchor. For a 736x1280 input:

```text
(736/8)  x (1280/8)  = 92 x 160 = 14,720
(736/16) x (1280/16) = 46 x 80  =  3,680
(736/32) x (1280/32) = 23 x 40  =    920
                                 19,320 predictions
```

Each prediction has 4 box numbers (centre x, centre y, width, height, in input pixels) and 65 class scores, so the raw output tensor is `[1, 69, 19320]`. Most rows score near zero everywhere.

## Two heads

YOLO26 is trained with two heads on the same features.

| Head | Trained to | Export output | Needs NMS |
|---|---|---|---|
| One-to-one (end-to-end) | Fire once per object | Top 300 rows of `[x1, y1, x2, y2, score, class]` | No |
| One-to-many | Let several nearby cells fire for one object | All raw predictions, `[4 + classes, anchors]` | Yes |

Ultralytics exports the one-to-one head by default (`end2end=True`). The served model uses the one-to-many head (`end2end=False`), so the code that reads its output removes duplicates itself.

## From raw output to boxes

1. Class: each prediction keeps its highest class score, and that score is its confidence.
2. Threshold: predictions below a confidence threshold are dropped. Thresholds can differ per class.
3. Box format: centre and size become corners, `x1 = cx - w/2` and so on.
4. Non-maximum suppression (NMS): sort by confidence, keep the best box, drop every other box of the same class that overlaps it too much, repeat. Class-wise NMS never lets a villager suppress a house.

Overlap is measured as intersection over union:

```text
IoU(A, B) = area(A ∩ B) / area(A ∪ B)
```

IoU is 0 for disjoint boxes and 1 for identical ones. The project uses 0.5 both as the NMS cut-off and as the bar for a prediction to match a labelled box.

## Input size and object scale

The network takes a fixed input size. Letterboxing resizes the image by one factor for both axes, so nothing is stretched, and pads the rest with grey `(114, 114, 114)`. Detections are mapped back by undoing the padding and the scale.

A detector learns objects at the pixel sizes it saw in training. A 3024x1672 capture letterboxed to 1280 wide shrinks by 0.42, and the model learned every sprite at that scale. Showing it the same frame at another scale moves objects outside what it learned, in either direction:

- Too small: thin objects and distant units fall below what the stride-8 grid can resolve.
- Too large: objects look unlike the training examples, and recall and precision both drop.

Sliced inference (SAHI) cuts a large image into tiles and runs each one at full input size, which enlarges small objects. It helps only when the enlarged objects match the training scale. The agent instead measures how large objects appear on screen and shrinks frames when they are too large (see [perception](../agent/perception.md)).

## Scoring a detector

On a labelled test set, each prediction is matched to at most one labelled box of the same class with IoU at or above the bar, highest confidence first.

| Count | Meaning |
|---|---|
| True positive (TP) | Prediction matched a label |
| False positive (FP) | Prediction matched nothing |
| False negative (FN) | Label matched nothing |

Precision is `TP / (TP + FP)`, recall is `TP / (TP + FN)`, and F1 is their harmonic mean. Raising a threshold usually trades recall for precision. Micro scores pool the counts over all classes, so common classes (trees, villagers) dominate them; per-class scores show the rare classes.

Benchmarks often report mAP, the area under the precision-recall curve averaged over classes, which summarises every possible threshold. The agent runs at one set of thresholds, so the project reports micro precision, recall and F1 at those thresholds.

## Training and fine-tuning

Training is supervised. The model sees images with labelled boxes, and a loss penalises box error and wrong class scores. Fine-tuning starts from weights pretrained on a general dataset and continues on game images, which needs far less data than training from scratch. The project's mix of real, synthetic and composite images is described in [training](../detection/training.md).

## Model files and runtimes

| File | Format | Used for |
|---|---|---|
| `.pt` | PyTorch checkpoint, loaded by Ultralytics | Training, fine-tuning, export |
| `.onnx` | Portable computation graph, run by onnxruntime | Serving. Runs on CPU, DirectML or CoreML through execution providers |
| `.mlpackage` | Apple CoreML model | Running directly on CPU and Neural Engine through `coremltools` |

An ONNX graph has a static or dynamic input shape. A static graph accepts exactly one shape, such as `[1, 3, 736, 1280]`, and some accelerators, including onnxruntime's CoreML provider at this size, accept only static graphs. A dynamic graph accepts varying batch and image sizes. Ultralytics writes the class names into the exported file's metadata, so a served model carries its own class list.
