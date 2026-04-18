# Denoiser Frontend

This is a standalone Next.js frontend for the `Denoiser` project.

## What it does

- accepts an uploaded image
- sends it to your deployed Hugging Face model through a server-side proxy route
- shows original and denoised previews
- lets the user download the output image

## Setup

1. Install Node.js 20 or later.
2. In this folder, install dependencies:

```bash
npm install
```

3. Copy the example env file:

```bash
cp .env.example .env.local
```

4. Set your Hugging Face endpoint in `.env.local`.

Minimum setup:

```env
HF_IMAGE_API_URL=https://your-endpoint-url-here
HF_API_TOKEN=hf_xxx_if_needed
HF_REQUEST_MODE=multipart
HF_IMAGE_FIELD_NAME=image
HF_EXTRA_FIELDS_JSON={}
```

## Run locally

```bash
npm run dev
```

Open `http://localhost:3000`.

## Env options

- `HF_IMAGE_API_URL`: required target URL for your deployed Hugging Face model
- `HF_API_TOKEN`: optional bearer token for private endpoints
- `HF_HTTP_METHOD`: defaults to `POST`
- `HF_REQUEST_MODE`:
  - `multipart` for endpoints that accept uploaded files
  - `raw` for endpoints that expect the image bytes directly
  - `json-base64` for endpoints that expect a base64 payload in `inputs`
- `HF_IMAGE_FIELD_NAME`: multipart field name, default `image`
- `HF_EXTRA_FIELDS_JSON`: optional JSON object of extra form or body values

## Notes

- The token stays server-side in the Next.js route handler.
- This scaffold cannot be runtime-tested here because Node.js is not installed in the current environment.
