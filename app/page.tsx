"use client";

import { ChangeEvent, useEffect, useState, useTransition } from "react";

type DenoiseState = {
  sourceFile: File | null;
  sourceUrl: string | null;
  resultUrl: string | null;
  status: string;
  error: string;
};

const initialState: DenoiseState = {
  sourceFile: null,
  sourceUrl: null,
  resultUrl: null,
  status: "Choose an image to start.",
  error: ""
};

export default function Page() {
  const [state, setState] = useState<DenoiseState>(initialState);
  const [isPending, startTransition] = useTransition();

  useEffect(() => {
    return () => {
      if (state.sourceUrl) {
        URL.revokeObjectURL(state.sourceUrl);
      }
      if (state.resultUrl) {
        URL.revokeObjectURL(state.resultUrl);
      }
    };
  }, [state.resultUrl, state.sourceUrl]);

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    if (!file) {
      return;
    }

    const nextSourceUrl = URL.createObjectURL(file);

    setState((current) => {
      if (current.sourceUrl) {
        URL.revokeObjectURL(current.sourceUrl);
      }
      if (current.resultUrl) {
        URL.revokeObjectURL(current.resultUrl);
      }

      return {
        sourceFile: file,
        sourceUrl: nextSourceUrl,
        resultUrl: null,
        status: `${file.name} is ready to denoise.`,
        error: ""
      };
    });
  }

  function clearSelection() {
    setState((current) => {
      if (current.sourceUrl) {
        URL.revokeObjectURL(current.sourceUrl);
      }
      if (current.resultUrl) {
        URL.revokeObjectURL(current.resultUrl);
      }
      return initialState;
    });
  }

  function submitForDenoise() {
    if (!state.sourceFile) {
      setState((current) => ({
        ...current,
        error: "Upload an image before trying to denoise it.",
        status: ""
      }));
      return;
    }

    const file = state.sourceFile;

    startTransition(async () => {
      const formData = new FormData();
      formData.append("image", file);

      setState((current) => ({
        ...current,
        error: "",
        status: "Processing your image..."
      }));

      try {
        const response = await fetch("/api/denoise", {
          method: "POST",
          body: formData
        });

        if (!response.ok) {
          const data = (await response.json().catch(() => null)) as
            | { error?: string; details?: string }
            | null;
          throw new Error(data?.details || data?.error || "The denoise request failed.");
        }

        const blob = await response.blob();
        const resultUrl = URL.createObjectURL(blob);

        setState((current) => {
          if (current.resultUrl) {
            URL.revokeObjectURL(current.resultUrl);
          }

          return {
            ...current,
            resultUrl,
            status: "Denoised image ready.",
            error: ""
          };
        });
      } catch (error) {
        setState((current) => ({
          ...current,
          error: error instanceof Error ? error.message : "Unexpected error during denoise.",
          status: ""
        }));
      }
    });
  }

  const downloadName = state.sourceFile
    ? state.sourceFile.name.replace(/\.(\w+)$/, "_denoised.png")
    : "denoised.png";

  return (
    <main className="shell">
      <section className="hero">
        <article className="heroCard">
          <div className="eyebrow">Image cleanup</div>
          <h1 className="title">Denoise your image in one clean pass.</h1>
          <p className="lede">
            Upload a noisy photo, clean it up instantly, compare the result side by side, and
            download the finished image when it looks right.
          </p>

          <div className="heroStats">
            <div className="stat">
              <strong>Upload</strong>
              <span>JPG, PNG, or any browser-supported image.</span>
            </div>
            <div className="stat">
              <strong>Process</strong>
              <span>Your image is processed securely behind the scenes.</span>
            </div>
            <div className="stat">
              <strong>Download</strong>
              <span>Save the denoised output instantly.</span>
            </div>
          </div>
        </article>

        <aside className="heroCard uploadCard">
          <label className="dropzone">
            <input type="file" accept="image/*" onChange={onFileChange} />
            <div className="dropzoneInner">
              <strong>{state.sourceFile ? state.sourceFile.name : "Drop an image here"}</strong>
              <p>
                {state.sourceFile
                  ? "Replace it anytime by choosing another file."
                  : "or click to browse from your device"}
              </p>
            </div>
          </label>

          <div className="actions">
            <button
              className="button buttonPrimary"
              type="button"
              onClick={submitForDenoise}
              disabled={isPending || !state.sourceFile}
            >
              {isPending ? "Denoising..." : "Generate denoised image"}
            </button>
            <button className="button buttonSecondary" type="button" onClick={clearSelection}>
              Reset
            </button>
            {state.resultUrl ? (
              <a className="button buttonSecondary" href={state.resultUrl} download={downloadName}>
                Download image
              </a>
            ) : null}
          </div>

          <div className={`status ${state.error ? "statusError" : ""}`}>
            {state.error || state.status}
          </div>
        </aside>
      </section>

      <section className="gallery">
        <article className="panel">
          <div className="panelHeader">
            <h2>Original</h2>
            <span>{state.sourceFile ? state.sourceFile.type || "image" : "waiting"}</span>
          </div>
          <div className="imageFrame">
            {state.sourceUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={state.sourceUrl} alt="Original upload preview" />
            ) : (
              <div className="emptyState">Your uploaded image preview will appear here.</div>
            )}
          </div>
        </article>

        <article className="panel">
          <div className="panelHeader">
            <h2>Denoised</h2>
            <span>{state.resultUrl ? "ready to download" : "waiting for output"}</span>
          </div>
          <div className="imageFrame">
            {state.resultUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={state.resultUrl} alt="Denoised output preview" />
            ) : (
              <div className="emptyState">
                The model output will appear here after the request finishes.
              </div>
            )}
          </div>
        </article>
      </section>
    </main>
  );
}
