import { redirect } from "next/navigation";
import { NexusDashboard } from "@/components/dashboard/nexus-dashboard";
import { hasOwnerSession } from "@/lib/auth";
import { getOverview } from "@/lib/repository";

export default async function Home() {
  if (!(await hasOwnerSession())) redirect("/login");
  const overview = await getOverview();
  return <NexusDashboard initialOverview={overview} />;
}

