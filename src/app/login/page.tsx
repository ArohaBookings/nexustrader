import { ShieldCheck } from "lucide-react";
import { loginAction } from "@/app/login/actions";
import { hasOwnerSession } from "@/lib/auth";
import { redirect } from "next/navigation";

export default async function LoginPage({
  searchParams,
}: {
  searchParams: Promise<{ error?: string }>;
}) {
  if (await hasOwnerSession()) redirect("/");
  const params = await searchParams;
  const missingConfig = !process.env.APP_ADMIN_PASSWORD || !process.env.SESSION_SECRET;

  return (
    <main className="grid min-h-screen place-items-center bg-[radial-gradient(circle_at_top_left,#164e63_0,transparent_32%),linear-gradient(135deg,#05070a_0%,#090f12_45%,#030406_100%)] px-5 text-white">
      <section className="w-full max-w-sm border border-cyan-400/20 bg-black/70 p-6 shadow-[0_0_60px_rgba(34,211,238,0.12)] backdrop-blur">
        <div className="mb-7 flex items-center gap-3">
          <div className="grid size-11 place-items-center border border-cyan-300/40 bg-cyan-300/10 text-cyan-200">
            <ShieldCheck size={23} />
          </div>
          <div>
            <h1 className="text-xl font-semibold tracking-[0.18em]">NEXUS TRADER</h1>
            <p className="text-xs uppercase tracking-[0.28em] text-cyan-200/70">Owner Access</p>
          </div>
        </div>
        <form action={loginAction} className="space-y-4">
          <label className="block text-sm text-zinc-300" htmlFor="password">
            Password
          </label>
          <input
            id="password"
            name="password"
            type="password"
            autoComplete="current-password"
            className="h-12 w-full border border-white/15 bg-zinc-950 px-4 text-base text-white outline-none transition focus:border-cyan-300"
          />
          {params.error ? <p className="text-sm text-red-300">Invalid owner password.</p> : null}
          {missingConfig ? (
            <p className="text-sm text-amber-200">Set APP_ADMIN_PASSWORD and SESSION_SECRET before using the dashboard.</p>
          ) : null}
          <button
            type="submit"
            className="h-12 w-full border border-cyan-300/50 bg-cyan-300 text-sm font-bold uppercase tracking-[0.18em] text-black transition hover:bg-cyan-200 disabled:opacity-50"
            disabled={missingConfig}
          >
            Enter
          </button>
        </form>
      </section>
    </main>
  );
}

