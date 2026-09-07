"""Canonical action-name taxonomy for usage events.

These constants are the shared vocabulary between every emitter
(agent-rfx, mcp-doc-qa, saas-admin) and every consumer (saas-admin's Usage
Dashboard) of usage events -- see ``core_lib.tracing.usage_events``. Keeping
them here, rather than as ad hoc string literals in each repo, is what lets
the dashboard's action filter and each service's ``log_usage_event()`` calls
stay in sync.

Two tiers exist:

- ``ACTION_*`` -- emitted by the service that actually performs the work
  (mcp-doc-qa, agent-rfx), once the action truly completes.
- ``ACTION_*_REQUESTED`` -- emitted by saas-admin at the point a user (or a
  system trigger) kicks the action off. Both tiers share the same
  ``process_id`` for one logical task, so the dashboard can show either the
  "user action" view (saas-admin rows) or the "API call" view (downstream
  service rows) for the same task, correlated via the existing log-chain
  drill-down.
"""

# Completion-side actions (emitted by the service doing the work)
ACTION_DOCUMENT_INGESTED = "document.ingested"
ACTION_RFX_INGESTED = "rfx.ingested"
ACTION_RFX_ANSWERED = "rfx.answered"
ACTION_RFX_VERIFIED = "rfx.verified"
ACTION_RFX_COMPARED = "rfx.compared"
ACTION_RFX_ANALYZED = "rfx.analyzed"
# Kept as the pre-existing event name used by mcp-doc-qa's answer-strategy
# logging rather than renamed, so this taxonomy registers it without
# breaking any existing consumer of that string.
ACTION_QUESTION_ANSWERED = "answer.strategy"

# Request-side actions (emitted by saas-admin at task origin)
ACTION_DOCUMENT_INGESTION_REQUESTED = "document.ingestion_requested"
ACTION_RFX_ANALYSIS_REQUESTED = "rfx.analysis_requested"
ACTION_RFX_PROCESSING_REQUESTED = "rfx.processing_requested"
ACTION_RFX_VERIFICATION_REQUESTED = "rfx.verification_requested"
ACTION_RFX_REFERENCE_INGESTION_REQUESTED = "rfx.reference_ingestion_requested"
ACTION_RFX_DELIVERY_CLEAN_REQUESTED = "rfx.delivery_clean_requested"
ACTION_RFX_BRIEF_GENERATION_REQUESTED = "rfx.brief_generation_requested"
ACTION_RFX_DOCUMENT_GENERATION_REQUESTED = "rfx.document_generation_requested"

__all__ = [
    "ACTION_DOCUMENT_INGESTED",
    "ACTION_RFX_INGESTED",
    "ACTION_RFX_ANSWERED",
    "ACTION_RFX_VERIFIED",
    "ACTION_RFX_COMPARED",
    "ACTION_RFX_ANALYZED",
    "ACTION_QUESTION_ANSWERED",
    "ACTION_DOCUMENT_INGESTION_REQUESTED",
    "ACTION_RFX_ANALYSIS_REQUESTED",
    "ACTION_RFX_PROCESSING_REQUESTED",
    "ACTION_RFX_VERIFICATION_REQUESTED",
    "ACTION_RFX_REFERENCE_INGESTION_REQUESTED",
    "ACTION_RFX_DELIVERY_CLEAN_REQUESTED",
    "ACTION_RFX_BRIEF_GENERATION_REQUESTED",
    "ACTION_RFX_DOCUMENT_GENERATION_REQUESTED",
]
