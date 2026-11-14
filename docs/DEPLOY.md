# Deploy & operate

Road Watch is a command-line scanner today (phase 0). "Deploy" means: a Docker
image, built and published automatically, that anyone can run on their own
footage. The Android app and the public hotspot map come in later phases.

## 1. How releases work

```
push to main ──► CI: tests ──► build image ──► smoke test ──► ghcr.io/kprsnt2/road-watch:latest  (+ :sha-xxxxxxx)
git tag v0.2.0 ─► same pipeline ─────────────────────────────► ghcr.io/kprsnt2/road-watch:0.2.0
pull request ──► tests + build + smoke test, nothing published
```

Workflow: `.github/workflows/ci.yml`. It uses the built-in `GITHUB_TOKEN`; no
secrets to set up.

### Package visibility

The image is linked to this public repo (via the `org.opencontainers.image.source`
label) and is publicly pullable without login. If a fork or a renamed repo ever
gets `denied` on `docker pull`, make the package public:
github.com/<owner> → **Packages** → `road-watch` → **Package settings** →
**Change visibility** → Public.

### Cutting a release

```bash
# bump version in pyproject.toml, commit, then:
git tag v0.2.0
git push origin v0.2.0
```

## 2. Running the image

```bash
docker pull ghcr.io/kprsnt2/road-watch:latest

# Put your files in one folder and mount it at /data.
docker run --rm -v "$PWD:/data" ghcr.io/kprsnt2/road-watch keygen device.key
docker run --rm -v "$PWD:/data" ghcr.io/kprsnt2/road-watch scan ride.mp4 \
    --gpx ride.gpx --start 2026-11-14T10:00:00+05:30 --key device.key --out out
docker run --rm -v "$PWD:/data" ghcr.io/kprsnt2/road-watch verify ride.mp4
```

Windows notes:

| Shell | Mount syntax |
|---|---|
| PowerShell | `-v "${PWD}:/data"` |
| cmd | `-v "%cd%:/data"` |
| Git Bash | prefix the command with `MSYS_NO_PATHCONV=1` and use `-v "$(cygpath -w "$PWD"):/data"` |

On Linux, add `--user "$(id -u):$(id -g)"` so output files belong to you.

## 3. Building locally

```bash
docker build -t road-watch:local .     # ~1 min after the first build; first build downloads PyTorch for the export stage
docker run --rm road-watch:local plate TG09EA1234
```

The first stage downloads PyTorch only to export the YOLO model to ONNX; the
final image contains just Python, ONNX Runtime, OpenCV and the model.

## 4. Swapping in your own models

Any YOLOv8 / YOLO11 model exported to ONNX works:

```bash
docker run --rm -v "$PWD:/data" ghcr.io/kprsnt2/road-watch scan ride.mp4 \
    --model models/my-coco-finetune.onnx --helmet-model models/helmet.onnx
```

Helmet models are expected to have a "no helmet" class; the common spellings
(`Without Helmet`, `without_helmet`, `head`, `no-helmet`, `NoHelmet`) are
mapped automatically.

## 5. What not to deploy publicly

- `events.jsonl`, key frames, videos, `*.evidence.json` contain plates and exact
  GPS. Keep them private. Only `hotspots.jsonl` is meant for a public map.
- `device.key` is the signing key. Never commit or share it; share only the
  public key printed by `keygen`.
- Model weights: YOLO11 is AGPL-3.0. Publishing the image publicly is fine for an
  open-source project; a closed-source product needs a different licence or detector.
