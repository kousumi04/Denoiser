# Denoiser

An end-to-end image denoising project built around DRUNet.

The repository covers:
- dataset preparation for SIDD and DIV2K noisy domains,
- DRUNet training and fine-tuning,
- local inference,
- a FastAPI backend packaged for Hugging Face Spaces,
- and a Next.js frontend deployed on Vercel.

## What The Project Does

The app takes a noisy image, estimates or accepts a noise level, runs it through DRUNet, and returns a cleaned PNG.

Current live services:
- Frontend: Vercel
- Backend: Hugging Face Space at `https://kousumi04-denoiser.hf.space`
- Denoise endpoint: `POST /denoise`

## Repository Layout

- `preprocessing_pipeline/` - prepares SIDD into paired training patches
- `data_pipeline/` - dataset loaders and pairing logic
- `training/` - DRUNet training script and model code
- `inference/` - local inference script
- `app.py` - FastAPI backend for Hugging Face Spaces
- `checkpoints/` - saved model weights
- `frontend/` - Next.js UI source
- `Dockerfile` - Hugging Face Docker Space definition

## Training Data Preparation

### SIDD

`preprocessing_pipeline/preprocess_sidd.py` reads the raw SIDD scene folders, finds the `NOISY_SRGB` and `GT_SRGB` images, crops paired patches, and saves them as `.pt` files.

Default output:
- `preprocessed_sidd/train/sample_XXXXXX.pt`

Each saved file contains:
- `noisy`: `C x H x W` tensor
- `clean`: `C x H x W` tensor

### DIV2K Noisy Domains

`data_pipeline/drunet_datasets.py` pairs clean DIV2K images with multiple noisy variants:
- Gaussian
- JPEG
- LowLight
- Poisson
- RandomCombo
- Speckle

The loader matches clean and noisy images by relative path and computes a per-sample `sigma_hat` from the noisy-clean difference.

## Training

`training/training_drunet.py` trains a unified DRUNet model over:
- DIV2K noisy domains
- preprocessed SIDD patches

Training details:
- optimizer: AdamW
- scheduler: cosine annealing
- loss: `L1 + 0.1 * Charbonnier`
- optional mixed precision
- gradient clipping enabled
- validation tracked with PSNR

Example:

```bash
python training/training_drunet.py --epochs 30 --batch-size 4 --patch-size 128
```

## Fine-Tuning And Resuming

The training script can resume from a checkpoint with `--resume`.

It supports both:
- full training checkpoints containing optimizer, scheduler, and scaler state
- plain `state_dict` checkpoints

This makes it easy to fine-tune from a previous run or continue a partially completed training job.

## Where The Model Is Saved

Training outputs are written to `checkpoints/`:

- `best_drunet_unified.pt` - best validation checkpoint during training
- `drunet_unified_final.pt` - final model weights saved at the end of training

The deployed Hugging Face Space uses:

- `checkpoints/DRUNET_DENOISER.pt`

That is the checkpoint loaded by `app.py` when the backend starts.

## Inference

For local inference, use:

```bash
python inference/infer_drunet.py path/to/image.png --checkpoint checkpoints/DRUNET_DENOISER.pt
```

If `--sigma` is not provided, the script estimates a noise level from the input image.

## Backend Deployment On Hugging Face Spaces

The backend is a FastAPI app defined in `app.py` and packaged with `Dockerfile`.

The Space:
- loads `checkpoints/DRUNET_DENOISER.pt` on startup
- exposes `GET /health`
- exposes `POST /denoise`
- accepts a multipart form upload with the file field named `image`
- returns a PNG image response

To deploy the backend:
1. Create or open the Hugging Face Space.
2. Use the Docker Space runtime.
3. Push the repo contents that include `app.py`, `Dockerfile`, `requirements-space.txt`, `training/`, and `checkpoints/`.
4. Make sure the Space has access to the checkpoint file `checkpoints/DRUNET_DENOISER.pt`.

## Frontend Deployment On Vercel

The Next.js frontend is configured to call the Hugging Face backend through the repo's Vercel config.

Key environment values:

```json
{
  "HF_IMAGE_API_URL": "https://kousumi04-denoiser.hf.space/denoise",
  "HF_HTTP_METHOD": "POST",
  "HF_REQUEST_MODE": "multipart",
  "HF_IMAGE_FIELD_NAME": "image"
}
```

To deploy the frontend:
1. Create a Vercel project from this GitHub repo.
2. Set the project to use the Next.js app in the repo root.
3. Deploy from `main`.
4. The frontend will proxy uploads to the Hugging Face `/denoise` endpoint.

## Local Backend Run

If you want to run the backend locally:

```bash
pip install -r requirements-space.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Notes

- The repo uses a single DRUNet checkpoint for the deployed backend.
- The training pipeline is designed to combine multiple noisy domains rather than a single dataset.
- The deployed frontend is only the UI layer; inference happens on the Hugging Face backend.

