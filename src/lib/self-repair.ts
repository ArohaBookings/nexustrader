import { executeBridgeControl } from "@/lib/bridge";
import { buildInstitutionalIntelligence } from "@/lib/institutional-intelligence";
import { createCommandRequest, getOverview, markCommandExecuted } from "@/lib/repository";

function repairId() {
  return `repair_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

export async function runSelfRepairAudit(source = "system") {
  const overview = await getOverview();
  const intelligence = buildInstitutionalIntelligence(overview);
  const repair = intelligence.selfRepair;
  const action = repair.recommendedBridgeAction;

  if (action !== "refresh_state") {
    return {
      ok: true,
      mode: "audit_only",
      action: "none",
      repair,
      reason: repair.hardRails.length ? "hard_rail_not_repairable" : repair.status,
    };
  }

  const commandId = repairId();
  await createCommandRequest({
    commandId,
    chatId: "system",
    action: "refresh_state",
    requestedText: "self-repair soft blocker refresh",
    confirmationRequired: false,
    payload: { source, repair },
  });
  const result = await executeBridgeControl("refresh_state");
  await markCommandExecuted(commandId, result);
  return {
    ok: Boolean(result.ok),
    mode: "self_repair",
    action: "refresh_state",
    commandId,
    result,
    repair,
  };
}
