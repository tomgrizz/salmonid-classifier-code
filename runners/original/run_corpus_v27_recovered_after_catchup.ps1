# ==================================================================================================
# HISTORICAL RECORD. NOT RUNNABLE FROM THIS REPOSITORY.
#
# Everything below this banner is the script exactly as it was on the authors' workstation.
# It is kept so that a reader can see which commands and settings produced the published
# results. It calls scripts by their names in the private research repository and uses that
# workstation's drive letters and folders, so it will not run anywhere else.
#
# Role in the published v2.7 run: waited for run_corpus_v27_track_catchup.ps1 to finish, then
# started detection and tracking of the recovered 2020 Ganaraska footage (2-3 August 2026).
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
# Wait for the tracking catch-up to drain, then run the recovered 2020 Ganaraska season.
#
# The two must not overlap: the catch-up re-decodes video for OSNet re-ID while the recovered pass
# runs detection, and on 2026-08-01 concurrent G: readers dropped the GPU from 99% to 0%.
#
# Waits on the catch-up's own completion line rather than on process names, so it cannot be fooled
# by an unrelated python process.
$repo = "C:\Users\lomu-\Documents\GitHub\lake-ontario-fish-cv"
$log  = "$repo\logs\corpus_v27_recovered.log"
$catchup = "$repo\logs\corpus_v27_track_catchup.log"
function Say($m) { "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $m" | Tee-Object $log -Append }

Say "waiting for the tracking catch-up to report COMPLETE"
$deadline = (Get-Date).AddHours(6)
while ((Get-Date) -lt $deadline) {
    if ((Test-Path $catchup) -and (Select-String -Path $catchup -Pattern 'catch-up COMPLETE' -Quiet)) {
        Say "catch-up complete; starting the recovered 2020 Ganaraska season"
        & powershell.exe -NoProfile -ExecutionPolicy Bypass `
            -File "$repo\runners\run_corpus_v27_backfill.ps1" -Only2020Recovered
        Say "recovered season stage returned"
        exit 0
    }
    Start-Sleep -Seconds 60
}
Say "TIMED OUT after 6 h waiting for the catch-up; recovered season NOT started"
exit 1
