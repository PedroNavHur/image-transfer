"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="navbar mx-auto w-full max-w-7xl px-4 py-3">
      <div className="flex-1">
        <Link href="/" className="btn btn-ghost text-xl">
          StyleForge
        </Link>
      </div>
      <ul className="menu menu-horizontal hidden gap-2 md:flex">
        <li>
          <Link href="/#features">Features</Link>
        </li>
        <li>
          <Link href="/#models">Models</Link>
        </li>
        <li>
          <Link href="/#pricing">Pricing</Link>
        </li>
        <li>
          <Link href="/#faq">FAQ</Link>
        </li>
      </ul>
      <div className="flex-none">
        {pathname !== "/studio" && (
          <Link href="/studio" className="btn btn-primary rounded-lg">
            Open Studio
          </Link>
        )}
      </div>
    </nav>
  );
}
