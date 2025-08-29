import Link from "next/link";

export function Footer() {
  return (
    <footer className="border-base-300 border-t">
      <div className="footer mx-auto w-full max-w-7xl p-6 text-sm">
        <aside>
          <span className="font-semibold">StyleForge</span>
          <p className="text-base-content/70">
            © {new Date().getFullYear()} — Made with ONNX Runtime Web + DaisyUI
          </p>
        </aside>
        <nav>
          <h6 className="footer-title">Links</h6>
          <Link className="link link-hover" href="/#features">
            Features
          </Link>
          <Link className="link link-hover" href="/#models">
            Models
          </Link>
          <Link className="link link-hover" href="/#pricing">
            Pricing
          </Link>
          <Link className="link link-hover" href="/#faq">
            FAQ
          </Link>
        </nav>
        <nav>
          <h6 className="footer-title">App</h6>
          <Link className="link link-hover" href="/studio">
            Open Studio
          </Link>
        </nav>
      </div>
    </footer>
  );
}
