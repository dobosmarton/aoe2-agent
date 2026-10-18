# Historical real-image audit for the next detector dataset

The v9 mixed dataset contains 219 distinct January 2026 real screenshots:
187 in its old training split and 32 in its old validation split. The training
files include 1,683 exact oversample copies. The 219 distinct images have
9,678 labeled boxes, all with present labels, valid class IDs, valid normalized
coordinates, and decodable images. Each has a matching source screenshot in
`real_screenshots/raw/`, but the labeled copy is a 1280×831 JPEG derived from
a 3024×1964 desktop capture. Use the labeled copy, not the raw PNG, because
its coordinates refer to the JPEG canvas.

No exact image hashes overlap between those 219 images and the 53 reviewed
October-batch images. A 64-bit whole-image perceptual-hash screen found no
old/new pair within Hamming distance eight. This is a duplicate screen, not a
proof of scene independence. The historical frames show the Mac/VM chrome and
were sampled at a different size and capture convention from the new native
game-window frames. A small label-overlay spot check found generally usable
boxes, but did not establish exhaustive annotation quality.

The old v9 split is unsuitable as validation for retraining: nine old
train/validation pairs had perceptual-hash distance at most eight, including
screenshots from the same minute only seconds apart. The preparation command
therefore discards the old split assignment and imports **all 219 distinct
images as training only**. It checks the old class names against the current
65-class schema, validates every label, verifies oversample copies, and
deduplicates them. The old 60 IDs retain their meaning in the new schema;
classes are remapped by name rather than assumed by position.

With `--oversample-real 4`, the prepared dataset contains 45 newly reviewed
and 219 historical real training images (1,056 files after oversampling),
2,400 synthetic training images, eight newly reviewed real development
validation images, and 600 synthetic images in a separate diagnostic split.
Eleven of the 65 classes still have no real boxes anywhere in this dataset.
The small development validation split is **not** a final benchmark; the
historical images are also not a held-out test. Before claiming improvement,
capture and manually review independent full-gameplay test sessions, freeze
the candidate and thresholds on development data, then score both models
once on that same test set. A training ablation excluding the historical
images is worthwhile because their desktop chrome and lower resolution may
hurt transfer to native game-window captures.
