# Annotation tools for the AoE2 detector review batch

Research date: 2026-10-12. This compares tools for reviewing **53 local
development screenshots, 1,814 draft bounding boxes, and 60 classes**. The
51-image main batch and 2-image scenario addendum already have separate COCO
files and image archives under
`packages/detection/src/real_screenshots/labeling_batch_20261011_*`.
These are model suggestions, **not reviewed ground truth**. Counts were read
from the two local `prelabels.coco.json` files.

## Recommendation

**Pilot X-AnyLabeling locally on the two scenario frames, then use it for the
51-image batch if import and round-trip validation pass.** Unlike the original
AnyLabeling app, the maintained X-AnyLabeling fork explicitly documents COCO
rectangle import and COCO/YOLO export; it has a macOS arm64 release and a CPU
installation option. Its local image/sidecar-JSON workflow avoids CVAT Online's
task and storage quotas and does not require a server. [X-AnyLabeling user
guide](https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/en/user_guide.md),
[installation guide](https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/en/get_started.md),
[macOS release](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v4.1.0).

The [COCO importer source](https://github.com/CVHub520/X-AnyLabeling/blob/main/anylabeling/views/labeling/label_converter.py)
maps `category_id` through the file's `categories`, converts `[x, y, w, h]`
into rectangle points, and matches images by basename. That structurally
matches our two COCO files (60 categories, basename `file_name`, bbox and empty
segmentation), but **we have not yet run an actual application import**. Its
importer does not preserve the draft `score` in the created shapes; keep the
original COCO and manifests separately for provenance. Its `checked` flag and
next-unchecked navigation give a useful review queue, but marking an image
checked must mean *all in-scope visible objects* were inspected, not merely
that the suggested boxes looked plausible. [Importer source](https://github.com/CVHub520/X-AnyLabeling/blob/main/anylabeling/views/labeling/label_converter.py),
[review and saving guide](https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/en/user_guide.md).

## Compared options

| Tool | Import current COCO boxes | Edit/review and export | Locality / operational cost | Fit |
| --- | --- | --- | --- | --- |
| **X-AnyLabeling** | Direct COCO rectangle import, with a `classes.txt` list; CLI also has `coco2xlabel`. [Guide](https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/en/user_guide.md), [CLI](https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/en/cli.md) | Desktop canvas supports moving/resizing/relabeling; checked status and next-unchecked navigation; COCO and YOLO detection export. [Guide](https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/en/user_guide.md) | Local app and per-image JSON, no hosted task quota. Separate application install; verify import/export mapping. [Guide](https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/en/user_guide.md), [install](https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/en/get_started.md) | **Best first pilot** |
| **Self-hosted CVAT Community** | Direct COCO 1.0 JSON import for bounding boxes with empty segmentation. [COCO format](https://docs.cvat.ai/docs/manual/advanced/formats/format-coco/) | Existing familiar web UI; exports COCO and YOLO. [COCO format](https://docs.cvat.ai/docs/manual/advanced/formats/format-coco/), [YOLO format](https://docs.cvat.ai/docs/manual/advanced/formats/format-yolo/) | Data can stay on the Mac, but Docker Compose starts multiple services including PostgreSQL and Redis; needs local storage, upgrades and backups. Mac instructions exist, though the page is old enough to title its section “Mac OS Mojave.” [Install](https://docs.cvat.ai/docs/administration/community/basics/installation/) | Best if familiar CVAT UI or collaboration outweighs maintenance |
| **Label Studio Community Edition** | Its native task/prediction JSON is not COCO; COCO needs the first-party converter or a small explicit transform, plus image URLs/local-file serving. [COCO converter in maintainer issue](https://github.com/HumanSignal/label-studio/issues/808), [pre-annotations](https://labelstud.io/guide/predictions), [local files](https://labelstud.io/guide/storage_local) | Rectangle labeling and COCO/YOLO exports; predictions can be distinguished from reviewed annotations. [Pre-annotations](https://labelstud.io/guide/predictions), [export](https://labelstud.io/guide/export) | Can install on macOS locally, but adds a server, project configuration and file-serving setup. The docs recommend roughly 100k tasks/annotations per project for performance, far above this batch—not a guarantee for our 60-label UI. [Install](https://labelstud.io/guide/install.html), [import guidance](https://labelstud.io/guide/tasks.html) | Capable, more conversion/setup than X-AnyLabeling |
| **Original AnyLabeling** | Its current official docs establish COCO/YOLO **export** and custom ONNX auto-labeling, but I did not find documented direct COCO annotation import in that project's docs. [Export](https://anylabeling.nrl.ai/docs/data-exporting), [custom models](https://anylabeling.nrl.ai/docs/custom-models) | Desktop rectangles and COCO/YOLO export. [Project README](https://github.com/vietanhdev/anylabeling), [export](https://anylabeling.nrl.ai/docs/data-exporting) | Local macOS build. Would require another conversion path for our prelabels unless a supported import is verified. [README](https://github.com/vietanhdev/anylabeling) | X-AnyLabeling is clearer for this exact import |
| **Labelme** | Native format is one JSON per image; the official README does not establish direct COCO import, so conversion is needed. [README](https://github.com/wkentaro/labelme) | Desktop rectangle editor; official COCO example targets instance segmentation, and native YOLO export is still an open request, so an additional converter is needed for this detection workflow. [README](https://github.com/wkentaro/labelme), [YOLO request](https://github.com/wkentaro/labelme/issues/2208) | Local on macOS; current v7 needs Python 3.12+, but it can run separately from the agent's Python 3.11 environment. [README](https://github.com/wkentaro/labelme) | Simple editor, poor current-format fit |
| **Make Sense** | Browser app explicitly lists COCO rectangle import. [Project README](https://github.com/SkalskiP/make-sense) | Rectangle editing and YOLO export. Its own format matrix does **not** list COCO rectangle export; use YOLO output or validate a converter. [README format matrix](https://github.com/SkalskiP/make-sense) | Official README says images stay in the browser, and it offers a local Node/Docker deployment. Browser-local work needs deliberate periodic exports/backups. [README](https://github.com/SkalskiP/make-sense) | Good zero-install experiment, weaker round trip |

No local tool has been load-tested with all 53 high-resolution frames and
1,814 boxes here. “No cloud quota” is a consequence of local storage, not a
claim of unlimited memory or performance. A two-frame import/export pilot is
the needed evidence before migrating the whole batch.

## Safe pilot and acceptance check

1. Keep the original image ZIPs, manifests and `prelabels.coco.json` files
   unchanged. Extract the two-image addendum into a *working copy*; do not
   point an annotator's auto-save at the only copy of the images.
2. In X-AnyLabeling, open that image directory and import the addendum COCO
   file as **rectangle/detection**, using `classes.txt` in the exact order of
   `packages/detection/src/training/config/classes.yaml`. The importer
   documents that class list requirement. [Guide](https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/en/user_guide.md).
3. Confirm the 2 images, 70 draft boxes and representative classes/locations
   (sheep, berry, villager, knight) visually. Move one box, change one label,
   delete one false positive and add one missed object. Save and reopen; verify
   all edits persist. [Editing and save behavior](https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/en/user_guide.md).
4. Export detection COCO and compare image basenames, category names and
   counts with the reviewed sidecar files. Expect IDs to change; compare names
   and geometry, not integer IDs. Also test YOLO export if that is the training
   handoff. [Export guide](https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/en/user_guide.md).
5. Only after the pilot passes, repeat in a separate working copy for the 51
   main frames. Review every visible in-scope object, especially classes the
   current model misses. Track each reviewed image with the app's checked
   status; archive final sidecar JSON/COCO export and preserve the original
   prediction file for audit.

If the X-AnyLabeling round trip fails, use self-hosted CVAT Community next:
it is the closest known workflow to the CVAT Online task that was blocked by
account quota. Do not delete any existing cloud task to make room until a
verified backup exists.

## Local pilot result (2026-10-12)

Installed `x-anylabeling-cvhub[cpu]==4.1.0` as an isolated `uv` tool on the
Apple Silicon Mac. The two-image pilot opened in the GUI with all 70 draft
rectangles. A CLI COCO → XLABEL → COCO round trip retained both image names,
all 60 category names, and all 70 boxes. One box that extended to the right
image edge lost one pixel of width on export; the source predictions remain
untouched. This establishes import and export compatibility, **not annotation
quality or save/edit persistence**.

A separate working copy at
`packages/detection/src/real_screenshots/labeling_batch_20261011_xanylabeling_review/`
contains all 53 images and 1,814 draft rectangles. X-AnyLabeling opened it
successfully and displayed the first image as `[Unchecked] [1/53]`. No image
has been marked reviewed and no corrected annotations are claimed. See the
working copy's `README.md` for resume and export commands.
