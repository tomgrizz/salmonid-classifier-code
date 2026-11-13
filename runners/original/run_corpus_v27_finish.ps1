# ==================================================================================================
# HISTORICAL RECORD. NOT RUNNABLE FROM THIS REPOSITORY.
#
# Everything below this banner is the script exactly as it was on the authors' workstation.
# It is kept so that a reader can see which commands and settings produced the published
# results. It calls scripts by their names in the private research repository and uses that
# workstation's drive letters and folders, so it will not run anywhere else.
#
# Role in the published v2.7 run: a wrapper that ran run_corpus_v27_backfill.ps1 (recovered 2020
# Ganaraska footage only) and then run_corpus_v27_track_catchup.ps1, 2 August 2026.
#
# Source: research repository commit d06535a (2026-08-05), "v2.7 corpus pipeline: runners, integrity guards, and the analysis scripts".
#
# To run the same step from this repository:
#     python -m salmonid.count.run_corpus --steps all --dry-run     (prints every command)
#
# Script names in the research repository and their equivalents in this repository:
#   scripts\_deimv2_detect_store.py       ->  python -m salmonid.detect.detect_store
#   scripts\_track_op_from_store.py       ->  python -m salmonid.detect.track_from_store
#   scripts\_corpus_countr_from_store.py  ->  python -m salmonid.count.corpus_countr_from_store
#   scripts\_build_canonical_results.py   ->  python -m salmonid.results.build_canonical_results
#   scripts\_audit_store_integrity.py     ->  python -m salmonid.detect.audit_store_integrity
#   scripts\_ntfy_send.py                 ->  salmonid.util.ntfy (off unless SALMONID_NTFY_TOPIC is set)
# ==================================================================================================
# Finish the v2.7 corpus: the recovered 2020 Ganaraska season (which the 2026-08-02 backfill lost
# to a wrong footage root) followed by the three tracking passes lost to the re-ID path bug.
#
# Both failures were the same shape and both are now fixed at source:
#   * recovered footage lives on D:\Ganaraska\..., not the F: mirror -> pass --footage-root "D:\"
#   * the OSNet re-ID weights are resolved against the repo root, not the working directory
# Verified by smoke test before this script was written.
#
# Order: the recovered season first (it needs the GPU, ~19 h), then the catch-up tracking (~1 h).
$repo = "C:\Users\lomu-\Documents\GitHub\lake-ontario-fish-cv"
$log  = "$repo\logs\corpus_v27_finish.log"
function Say($m) { "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $m" | Tee-Object $log -Append }

Say "=== v2.7 finish BEGIN: recovered 2020 Ganaraska, then tracking catch-up ==="
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$repo\runners\run_corpus_v27_backfill.ps1" -Only2020Recovered
Say "--- recovered season stage returned; starting tracking catch-up ---"
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$repo\runners\run_corpus_v27_track_catchup.ps1"
Say "=== v2.7 finish COMPLETE ==="
