# Hugging Face options for the AoE2 detector

Research date: 2026-09-29

Release status (2026-09-29): v9 was uploaded to the private model repository
[`martondobos/aoe2-entity-detector`](https://huggingface.co/martondobos/aoe2-entity-detector).
The original `v9` tag remains at commit `424c9f1b62c20e3ec2f6586493742209ace3678d`
with the versioned filename. The current release uses stable `model.onnx` at
commit `684777fbb7a689a5ca866a4c8ed1e9ec5869bb12`, tagged `v9.1`; the
older path was removed from the current branch but remains recoverable from
`v9`. The model card was made version-neutral in the current branch at
commit `e6da6316760aab0c71173ee7dbb5d5ba0f9c4dee`; the `v9.1` tag remains
unchanged. A pinned download of the new ONNX and class schema matched the source
SHA-256 hashes. The original downloaded model also produced detections on a
real-frame fixture through the CPU ONNX Runtime path. The application still
serves its existing local model; no runtime download or hosted inference was
enabled.

## Recommendation

**Yes for versioning and sharing the model; not as the default live inference location.** After reviewing artifact rights, publish the current detector to a **private Hugging Face model repository first**, with a model card, class schema, inference contract, and real-screenshot evaluation. Keep the Windows VM → Mac detector server on the local link for gameplay. Consider a public repository only after reviewing public-distribution rights and license obligations. Consider hosted inference only if access from machines outside the local network or a public demo becomes a requirement.

The present gameplay path uses `aoe2_yolo_v9.onnx` (about 9.9 MB) at `imgsz=1280`, with the detector server's `POST /detect` accepting an image upload and returning bounding boxes; the VM-side client adds tracking. See the local [detector architecture](../part3-entity-detection/07-detector-architecture.md), [server](../../apps/detection-server/src/app.py), and [client](../../packages/detection/src/inference/remote_detector.py). A Hub model repository stores/version-controls artifacts; it does **not** itself run this custom HTTP contract. Hugging Face explicitly supports uploading custom models, including ones outside Transformers, but serving custom inference requires a handler or container. [Uploading models](https://huggingface.co/docs/hub/models-uploading), [custom handlers](https://huggingface.co/docs/inference-endpoints/guides/custom_handler), [custom containers](https://huggingface.co/docs/inference-endpoints/guides/custom_container).

## Options

| Option | Fit here | Operational implication |
| --- | --- | --- |
| **Model repository only** | Best immediate step: reproducible weights and metadata, controlled collaboration. | Create a private model repo and upload the ONNX artifact plus `classes.yaml` and a model card. Pin its commit revision when fetching; prefetch to the Mac at setup time so game runs do not depend on Hub availability. Private repos are invisible to others and require authorized access. [Repository setup](https://huggingface.co/docs/hub/repositories-getting-started), [visibility](https://huggingface.co/docs/hub/repositories-settings), [revision-pinned downloads](https://huggingface.co/docs/huggingface_hub/guides/download). |
| **Inference Endpoint** | Appropriate if several remote clients need a managed, authenticated detection API. | The ONNX preprocessing, postprocessing, and response schema need a custom handler or container. Dedicated compute is billed while initializing/running; scale-to-zero saves idle cost but adds cold starts. A token-protected endpoint is supported, with PrivateLink as a further network option. Measure full screenshot upload + queue + inference + response time against the current local path before switching. [Custom handler](https://huggingface.co/docs/inference-endpoints/guides/custom_handler), [custom container](https://huggingface.co/docs/inference-endpoints/guides/custom_container), [pricing](https://huggingface.co/docs/inference-endpoints/pricing), [autoscaling](https://huggingface.co/docs/inference-endpoints/guides/autoscaling), [configuration](https://huggingface.co/docs/inference-endpoints/guides/configuration). |
| **Docker Space** | Good for an interactive visual demo or occasional API experiment, not the first choice for the time-sensitive game loop. | It can host FastAPI, but free hardware sleeps; paid hardware is billed for running time and can also sleep if configured. Public, protected, and private Space visibility differ: a protected Space hides source while its app URL remains public, whereas a private Space restricts both. [Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker), [Space visibility and lifecycle](https://huggingface.co/docs/hub/spaces-overview), [hardware billing](https://huggingface.co/docs/hub/spaces-gpus). |

Do not assume that putting the model on the Hub makes it available through a shared Inference Provider. The [provider API](https://huggingface.co/docs/inference-providers/index) routes among models providers actually serve; a custom model with this project's ONNX decoder and response format needs an explicit serving implementation. This is an inference from the provider and custom-handler documentation, not a claim that object detection itself is unsupported.

## Registry choice for the current artifact

The distribution decision is separate from hosted inference. The current served file is a 9.9 MB ONNX model; the PyTorch checkpoint is about 5.3 MB. Both are ignored by the application Git repository. The detector also depends on the 60 ordered class IDs, 1280-pixel preprocessing, and the project's YOLO26 output decoder, so a bare weight file is not a reproducible release. [Local detector contract](../part3-entity-detection/07-detector-architecture.md).

| Registry | Strength | Drawback for this project |
| --- | --- | --- |
| **Hugging Face model repository (recommended)** | Purpose-built model card and license metadata; public or private visibility; arbitrary custom model files including ONNX; revision-pinned, cached downloads. [Model upload](https://huggingface.co/docs/hub/models-uploading), [cards](https://huggingface.co/docs/hub/model-cards), [downloads](https://huggingface.co/docs/huggingface_hub/guides/download). | A private repository needs a separate Hugging Face read credential on the Mac. Upload does not provide the custom detector server or make the model available through an Inference Provider. [Access tokens](https://huggingface.co/docs/hub/security-tokens). |
| **GitHub Release asset** | Simple binary distribution alongside a code tag; a 9.9 MB asset is below the per-file 2 GiB limit. [GitHub releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases). | Visibility follows the source repository, and there is no model-specific card or download helper. An asset or tag can be changed unless immutable releases are enabled; a checksum still needs to be recorded. [Release management](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository). |
| **Git LFS in the code repository** | Familiar Git workflow. | Every code checkout inherits model-file pointers and LFS download/storage accounting; GitHub recommends LFS for binaries but it adds quotas and setup to the agent checkout. Not useful when the Mac server alone needs the weights. [GitHub repository limits](https://docs.github.com/en/repositories/creating-and-managing-repositories/repository-limits), [LFS billing](https://docs.github.com/en/billing/concepts/product-billing/git-lfs). |
| **OCI artifact, e.g. GHCR via ORAS** | Can package weights and metadata as layers and pull by content digest; GHCR supports OCI and independently configurable private visibility. [ORAS push/pull](https://oras.land/docs/how_to_guides/pushing_and_pulling/), [GHCR](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry). | Adds ORAS tooling, package credentials, and artifact conventions without providing a model card or a benefit to the present one-model Mac setup. GHCR private access currently requires GitHub package authentication. [GHCR authentication](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry). |

Choose a **private Hub model repo** named `martondobos/aoe2-entity-detector`; keep the current local detector server and cache the pinned artifact before game launch. The repository name should outlive v9: use a `v9` tag and full commit revision for this release, then add future versions without changing the repository address. A public Hub repo is a later visibility change, contingent on artifact and license review. Hugging Face documents both private visibility and custom/non-Transformers model uploads. [Repository visibility](https://huggingface.co/docs/hub/repositories-settings), [model uploads](https://huggingface.co/docs/hub/models-uploading).

### Reproducible artifact manifest

Publish the following together in one model-repository commit (file names illustrative except for the existing weight and class-schema names):

| File | Contents |
| --- | --- |
| `model.onnx` | Exact v9 ONNX weight consumed by the Mac detector, under a stable registry filename. Current local SHA-256: `515a018bc2190fdf5427a01ff21e294331324929c8603d870c18255626cee8fd`. Recompute immediately before upload. |
| `classes.yaml` | The ordered 60-class ID/name schema from `packages/detection/src/training/config/classes.yaml`; the server-bundled copy currently differs only in header comments. Current local SHA-256: `5365dcf538d16f9b237070a5e9c7609028314794dd8e544233eecb76a09de717`. Recompute immediately before upload. |
| `inference-contract.json` | Model architecture/export, input size and color/normalization/letterboxing convention, ONNX input/output names and shape, class order, output decoder version or source commit, box-coordinate convention, default confidence/NMS settings, and expected ONNX Runtime version. Values should be extracted from the actual server code and export, not guessed. |
| `README.md` | Model card: intended use, training-data provenance, real-screenshot evaluation (including weak classes and exact split), limitations, version lineage, and carefully verified license metadata. Do not declare an MIT/Apache license until rights are established. [Model cards](https://huggingface.co/docs/hub/model-cards). |

Record in the application deployment configuration or run manifest: Hub `repo_id`, **full-length commit SHA** (not a moving `main` branch or short SHA), model-file SHA-256, class-schema SHA-256, and detector-server source revision. Hugging Face's `hf_hub_download`/`snapshot_download` `revision` accepts a commit hash, and the docs require the full hash for that form. Fetch and checksum both files during setup, then run from local cache so registry/network availability cannot pause gameplay. [Revision-pinned downloads](https://huggingface.co/docs/huggingface_hub/guides/download), [download API](https://huggingface.co/docs/huggingface_hub/package_reference/file_download).

The recommended upload sequence is: clear rights and verify checksums; create a private model repo; upload an explicit staging directory containing only the four release files; capture the returned commit SHA; download that revision into a clean cache; verify checksums and run a real-frame detector smoke test before changing deployment configuration. `hf repos create ... --private`, `hf upload`, and the `HfApi` upload methods are documented; the latter can upload arbitrary files and folders. Avoid uploading the entire training directory or screenshots by default. [Repository creation](https://huggingface.co/docs/huggingface_hub/guides/repository), [upload API/CLI](https://huggingface.co/docs/huggingface_hub/guides/upload).

## Publication checklist

1. Upload a pinned artifact set: `model.onnx`, matching `classes.yaml`, the 1280-pixel preprocessing/box-coordinate convention, and a model card stating intended AoE2:DE screenshot use. Keep training screenshots, extracted sprites, credentials, and experiment logs out unless each is deliberately cleared for release. The Hub supports model cards that document datasets, intended use, limitations, and evaluation. [Model cards](https://huggingface.co/docs/hub/model-cards), [uploading](https://huggingface.co/docs/hub/models-uploading).
2. Record evaluation on **real screenshots**, not only synthetic validation. The repo's measured v9 real-image micro-F1 is about 0.67 on its documented split; the model card should name the exact split, inference path, image size, and weak classes rather than implying general game competence. [Local evaluation notes](../part3-entity-detection/07-detector-architecture.md).
3. Review the rights to upload artifacts derived from game screenshots/extracted sprites even to a private third-party host, and verify the Ultralytics model/weight license before choosing a public license. Microsoft's permissions page directs game-content questions to its Game Content Usage Rules; it does not itself grant a license to publish a trained detector or dataset. Ultralytics states that its trained YOLO models are AGPL-3.0 by default or subject to its enterprise license; do not label the weights MIT/Apache merely because this repository might use another license. This is a release gate, not legal advice. [Microsoft copyrighted-content guidance](https://www.microsoft.com/legal/intellectualproperty/copyright/permissions), [Ultralytics licensing](https://www.ultralytics.com/license), [Hub license metadata](https://huggingface.co/docs/hub/model-cards#specifying-a-license).
4. For a private repo, give the Mac setup a fine-grained, read-only token scoped to that model; do not embed it in the VM agent, commit it, or print it in logs. Hugging Face recommends fine-grained per-application tokens. [Access tokens](https://huggingface.co/docs/hub/security-tokens). A private model repo is distinct from a gated public model: gating can grant access to other users on request. [Gated models](https://huggingface.co/docs/hub/models-gated).

## Decision gate for hosted inference

Benchmark the existing local server and a prospective hosted deployment using the same native frames and exact decoder. Compare end-to-end p50/p95 latency, timeout/error rate, detection F1, and monthly compute cost. Network upload and cold start make a remote service a likely regression for a VM and Mac on the same machine; that is an architectural inference, not a measured result. Remote inference also sends every detection frame to a third party. Hugging Face says Endpoint traffic is TLS-encrypted, does not retain request payloads, and retains logs for 30 days; assess whether that meets the project's data requirements. If remote access wins on an actual requirement, keep the same `/detect` contract or add an adapter, pin the model revision, use authenticated access, and disable scale-to-zero for runs that require predictable first-request latency. Hugging Face documents the [cold-start trade-off](https://huggingface.co/docs/inference-endpoints/guides/autoscaling), [revision selection](https://huggingface.co/docs/inference-endpoints/guides/advanced), and [endpoint security](https://huggingface.co/docs/inference-endpoints/guides/security).

## Registry best-practice audit (2026-09-30)

The private release already has the appropriate minimal artifact set: stable
`model.onnx`, ordered `classes.yaml`, `inference-contract.json`, and a model
card. Its `object-detection` task tag, intended-use statement, input/output
contract, caveats, and checksums are useful. No additional weights are needed
for the current Mac-hosted inference path. Hugging Face's [model-card guidance](https://huggingface.co/docs/hub/model-cards)
and [release checklist](https://huggingface.co/docs/hub/model-release-checklist)
emphasize reproducible use, data provenance, measured evaluation, and limits;
they do not require publishing training data or a hosted demo.

Priorities before treating this as a reproducible model release:

1. **Substantiate the real-frame score.** The card currently cites about 0.67
   F1 without the exact evaluated checkpoint, split manifest/size, IoU and
   confidence settings, per-class counts, or a raw-ONNX-runtime result file.
   The local, ignored `training_data_v9_slim/eval_real_summary.json` has only a
   `synth` result (600 images, F1 0.386); it must **not** be uploaded as evidence
   for the real-frame 0.67 figure. Re-run or recover the deployment-path real
   evaluation, verify a held-out split, and then publish an aggregate,
   machine-readable result plus a small method/results table in the card.
   Report the weak military classes and map/UI-scale limitations. The [model-card
   guidance](https://huggingface.co/docs/hub/model-card-annotated) calls for
   dataset, training, evaluation, and limitations detail; the [release
   checklist](https://huggingface.co/docs/hub/model-release-checklist) calls for
   measured performance rather than an unsupported benchmark claim. Do not
   publish private evaluation screenshots to make the score verifiable.
2. **Add a tested usage path.** The contract describes tensors, but the card
   has no copy-and-run example that downloads one full-commit-pinned snapshot,
   verifies `model.onnx` and `classes.yaml`, letterboxes an image, runs ONNX
   Runtime, and maps boxes to original pixels. A short example can instead
   invoke the version-pinned detector server, provided that it genuinely
   reproduces these steps and is tested in a clean environment. State the
   tested Python/ONNX Runtime versions and hardware/backend; distinguish raw
   model output from the server's thresholds and VM-side tracking. Hugging Face
   recommends executable usage examples and technical specifications in its
   [release checklist](https://huggingface.co/docs/hub/model-release-checklist),
   and supports [full-commit revision downloads](https://huggingface.co/docs/huggingface_hub/guides/download).
3. **Expand provenance without distributing source images.** Record approximate
   counts and sources for synthetic sprite composites and annotated real
   screenshots, collection/labeling method, train/validation separation,
   major game/UI conditions, model initialization, training resolution, and
   the source revision used for export. Add Hub `datasets` metadata only if a
   real Hub dataset with cleared rights exists. The current card's
   `library_name: ultralytics` describes the training/export origin; make clear
   that the shipped inference runtime is ONNX Runtime rather than implying a
   generic hosted Ultralytics pipeline. Hugging Face documents [dataset and
   library metadata](https://huggingface.co/docs/hub/model-cards) separately
   from the human-readable model-card detail.
4. **Resolve license metadata deliberately.** The card correctly avoids
   assigning a permissive license while game-derived artifact rights and
   Ultralytics obligations are unresolved. Once reviewed, publish the accurate
   license or custom license name/link in YAML and, if necessary, a `LICENSE`
   file. Until then, retain the private visibility and explicit rights caveat;
   do not infer redistribution rights from the ONNX file alone. Hugging Face
   documents [license metadata and custom licenses](https://huggingface.co/docs/hub/model-cards#specifying-a-license).

Keep the model repository private and pin consumers to **the full commit that
contains both the weights and the current card**, not a moving branch or an
older release tag. Use a separate, fine-grained read-only token on the Mac;
the upload credential should not be installed in the VM. Hugging Face documents
[revision-pinned downloads](https://huggingface.co/docs/huggingface_hub/guides/download),
[private visibility](https://huggingface.co/docs/hub/repositories-settings), and
[fine-grained token permissions](https://huggingface.co/docs/hub/security-tokens).

Do **not** add the `.pt` training checkpoint, extracted game sprites, training
screenshots, raw run logs, or an unreviewed validation dataset merely to make
the repository look complete. A separate dataset repo and card are appropriate
only if the images and annotations can be lawfully shared and the dataset's
provenance and limitations can be documented. A demo Space or browser widget
is likewise optional and would add deployment/privacy work without helping the
current Mac detector. See Hugging Face's [dataset-card guidance](https://huggingface.co/docs/hub/datasets-cards)
and [model-widget requirements](https://huggingface.co/docs/hub/models-widgets).
