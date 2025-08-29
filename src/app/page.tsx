// src/app/page.tsx
import Link from "next/link";

export default function LandingPage() {
  return (
    <main className="bg-base-200 relative min-h-dvh overflow-hidden">
      {/* soft gradient blobs */}
      <div className="pointer-events-none absolute inset-0 -z-10">
        <div className="bg-primary/20 absolute -top-40 -left-40 h-[28rem] w-[28rem] rounded-full blur-3xl" />
        <div className="bg-secondary/20 absolute -right-40 -bottom-40 h-[30rem] w-[30rem] rounded-full blur-3xl" />
      </div>

      {/* HERO */}
      <header className="mx-auto w-full max-w-7xl px-4 pt-6 pb-10 md:pt-12">
        <div className="grid items-center gap-8 md:grid-cols-2">
          <div className="space-y-6">
            <div className="badge badge-primary badge-outline">
              WebAssembly + ONNX Runtime
            </div>
            <h1 className="text-4xl leading-tight font-extrabold md:text-5xl">
              Transform photos into{" "}
              <span className="text-primary">stylized art</span>—right in your
              browser.
            </h1>
            <p className="text-base-content/70">
              Zero uploads. Private by design. AnimeGANv3 and Fast Neural Style
              running locally with ONNX Runtime Web. Dial the{" "}
              <b>style strength</b> to taste and download instantly.
            </p>

            <div className="flex flex-wrap gap-3">
              <Link href="/studio" className="btn btn-primary rounded-lg">
                Start Free
              </Link>
              <a href="#how" className="btn btn-outline rounded-lg">
                How it works
              </a>
            </div>

            <div className="stats stats-vertical md:stats-horizontal mt-4 w-full shadow">
              <div className="stat">
                <div className="stat-title">Privacy</div>
                <div className="stat-value text-primary">100%</div>
                <div className="stat-desc">In-browser processing</div>
              </div>
              <div className="stat">
                <div className="stat-title">Latency</div>
                <div className="stat-value">~100ms–1s</div>
                <div className="stat-desc">Device & model dependent</div>
              </div>
              <div className="stat">
                <div className="stat-title">Control</div>
                <div className="stat-value">Strength</div>
                <div className="stat-desc">Blend original vs. style</div>
              </div>
            </div>
          </div>

          {/* Mocked card preview */}
          <div className="card border-base-300 bg-base-100/80 border shadow-xl backdrop-blur">
            <div className="card-body gap-4">
              <h3 className="card-title text-base">Before / After</h3>
              <div className="grid gap-3 md:grid-cols-2">
                <div className="rounded-box border-base-300 bg-base-200 border p-2">
                  <img
                    src="https://img.daisyui.com/images/stock/photo-1559181567-c3190ca9959b.webp"
                    alt="Original example"
                    className="mx-auto aspect-video w-full rounded-md object-cover"
                  />
                </div>
                <div className="rounded-box border-base-300 bg-base-200 border p-2">
                  <img
                    src="https://img.daisyui.com/images/stock/photo-1560717789-0ac7c58ac90a-blur.webp"
                    alt="Stylized example"
                    className="mx-auto aspect-video w-full rounded-md object-cover"
                  />
                </div>
              </div>
              <div className="flex items-center justify-between">
                <div className="badge badge-ghost">AnimeGANv3</div>
                <Link
                  href="/studio"
                  className="btn btn-secondary btn-sm rounded-lg"
                >
                  Try this style
                </Link>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* HOW IT WORKS */}
      <section id="how" className="mx-auto w-full max-w-7xl px-4 py-10">
        <div className="text-center">
          <h2 className="text-3xl font-bold">How it works</h2>
          <p className="text-base-content/70 mt-2">
            Three simple steps—no sign-in needed.
          </p>
        </div>
        <ul className="steps steps-vertical md:steps-horizontal mx-auto mt-8 max-w-3xl text-sm">
          <li className="step step-primary">
            Upload an image (processed entirely in your browser).
          </li>
          <li className="step step-primary">
            Pick a style (AnimeGANv3 or Fast Neural Style).
          </li>
          <li className="step step-primary">
            Adjust strength & download your PNG.
          </li>
        </ul>
      </section>

      {/* FEATURES */}
      <section id="features" className="mx-auto w-full max-w-7xl px-4 py-10">
        <div className="grid gap-6 md:grid-cols-3">
          <FeatureCard
            emoji="🛡️"
            title="Private by default"
            text="All stylization happens locally with ONNX Runtime Web (WASM). No uploads."
          />
          <FeatureCard
            emoji="⚡"
            title="Fast & responsive"
            text="WebAssembly + optimized preprocessing keeps things snappy on modern devices."
          />
          <FeatureCard
            emoji="🎛️"
            title="Style strength"
            text="Blend the stylized output with the original for gentle or bold looks."
          />
        </div>
      </section>

      {/* MODELS */}
      <section id="models" className="mx-auto w-full max-w-7xl px-4 py-10">
        <div className="card border-base-300 bg-base-100 border shadow-xl">
          <div className="card-body">
            <h3 className="card-title">Supported Models</h3>
            <p className="text-base-content/70">
              Hand-picked ONNX models that run great in the browser.
            </p>

            <div className="mt-4 grid gap-6 md:grid-cols-2">
              <div>
                <h4 className="mb-2 text-sm font-semibold">Studio Look</h4>
                <div className="flex flex-wrap gap-2">
                  <span className="badge badge-outline">Ghibli</span>
                  <span className="badge badge-outline">Hayao</span>
                  <span className="badge badge-outline">Shinkai</span>
                  <span className="badge badge-outline">Cyberpunk</span>
                  <span className="badge badge-outline">Sketch</span>
                </div>
              </div>
              <div>
                <h4 className="mb-2 text-sm font-semibold">Filter Style</h4>
                <div className="flex flex-wrap gap-2">
                  <span className="badge badge-outline">Candy</span>
                  <span className="badge badge-outline">Mosaic</span>
                  <span className="badge badge-outline">Udnie</span>
                  <span className="badge badge-outline">Pointilism</span>
                </div>
              </div>
            </div>

            <div className="mt-6">
              <Link href="/studio" className="btn btn-primary rounded-lg">
                Open Studio
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* PRICING */}
      <section id="pricing" className="mx-auto w-full max-w-7xl px-4 py-10">
        <div className="grid gap-6 md:grid-cols-2">
          <div className="card border-base-300 bg-base-100 border shadow-xl">
            <div className="card-body">
              <h3 className="card-title">Free</h3>
              <p className="text-base-content/70">
                In-browser stylization with our ONNX presets.
              </p>
              <ul className="mt-2 space-y-1 text-sm">
                <li>• AnimeGANv3 + Fast Neural Style</li>
                <li>• Style strength slider</li>
                <li>• Privacy-first (no uploads)</li>
              </ul>
              <div className="card-actions mt-4">
                <Link href="/studio" className="btn btn-primary rounded-lg">
                  Start Free
                </Link>
              </div>
            </div>
          </div>

          <div className="card border-base-300 bg-base-100 border shadow-xl">
            <div className="card-body">
              <h3 className="card-title">Pro (Optional)</h3>
              <p className="text-base-content/70">
                Server-side heavy models (e.g., Flux, SDXL variants) via
                provider APIs.
              </p>
              <ul className="mt-2 space-y-1 text-sm">
                <li>• Higher fidelity & variety</li>
                <li>• Image-to-Image pipelines</li>
                <li>• Keep local mode as default</li>
              </ul>
              <div className="card-actions mt-4">
                <Link href="/studio" className="btn btn-outline rounded-lg">
                  Contact
                </Link>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* FAQ */}
      <section id="faq" className="mx-auto w-full max-w-7xl px-4 py-10">
        <h3 className="mb-4 text-2xl font-bold">FAQ</h3>
        <div className="join join-vertical w-full">
          <details className="join-item collapse-arrow border-base-300 bg-base-100 collapse border">
            <summary className="collapse-title text-base font-medium">
              Do images ever leave my device?
            </summary>
            <div className="collapse-content text-base-content/70 text-sm">
              No. All ONNX inference runs in your browser using WebAssembly.
              Downloads are generated locally.
            </div>
          </details>

          <details className="join-item collapse-arrow border-base-300 bg-base-100 collapse border">
            <summary className="collapse-title text-base font-medium">
              Which browsers are supported?
            </summary>
            <div className="collapse-content text-base-content/70 text-sm">
              Modern Chromium, Firefox, and Safari on desktop. Mobile works but
              large images may be memory-constrained.
            </div>
          </details>

          <details className="join-item collapse-arrow border-base-300 bg-base-100 collapse border">
            <summary className="collapse-title text-base font-medium">
              Can I fine-tune styles?
            </summary>
            <div className="collapse-content text-base-content/70 text-sm">
              Not yet in-app. You can adjust strength, pick alternate models, or
              use optional server styles for variety.
            </div>
          </details>
        </div>
      </section>
    </main>
  );
}

function FeatureCard({ emoji, title, text }: { emoji: string; title: string; text: string; }) {
  return (
    <div className="card border-base-300 bg-base-100 border shadow-xl">
      <div className="card-body">
        <div className="flex items-center gap-3">
          <div className="bg-base-200 grid h-10 w-10 place-items-center rounded-lg text-xl">
            <span aria-hidden>{emoji}</span>
          </div>
          <h3 className="card-title text-base">{title}</h3>
        </div>
        <p className="text-base-content/70">{text}</p>
      </div>
    </div>
  );
}
