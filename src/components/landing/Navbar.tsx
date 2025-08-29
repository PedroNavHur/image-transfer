import Link from "next/link";

export function Navbar() {
  return (
    <nav className="navbar mx-auto w-full max-w-7xl px-4 py-3">
      <div className="flex-1">
        <Link href="/" className="btn btn-ghost text-xl">
          StyleForge
        </Link>
      </div>
      <ul className="menu menu-horizontal hidden gap-2 md:flex">
        <li>
          <a href="#features">Features</a>
        </li>
        <li>
          <a href="#models">Models</a>
        </li>
        <li>
          <a href="#pricing">Pricing</a>
        </li>
        <li>
          <a href="#faq">FAQ</a>
        </li>
      </ul>
      <div className="flex-none">
        <Link href="/studio" className="btn btn-primary rounded-lg">
          Open Studio
        </Link>
      </div>
    </nav>
  );
}
