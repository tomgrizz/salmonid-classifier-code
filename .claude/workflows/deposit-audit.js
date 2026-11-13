export const meta = {
  name: 'deposit-audit',
  description: 'Read-only reproducibility audit of the deposit: review each area, adversarially verify every finding, confirm Tier 1 still passes',
  phases: [
    { title: 'Baseline', detail: 'run python run_tier1.py and record the result' },
    { title: 'Review', detail: 'one reviewer per area of the repository' },
    { title: 'Verify', detail: 'each finding is checked by an agent told to refute it' },
    { title: 'Synthesize', detail: 'merge confirmed findings against the existing register' },
  ],
}

// ---------------------------------------------------------------------------------------
// What this workflow is for
//   A second, independent pass over the deposit before submission. It changes nothing: every
//   agent is told to read and run, never to edit. Its output is a list of confirmed findings
//   that are NOT already recorded in docs/REPRODUCIBILITY_REVIEW.md.
//
// How to run
//   /deposit-audit                         audit every area listed in AREAS
//   /deposit-audit ["salmonid/count"]      audit only the named areas (args = array of paths)
//
// CLAUDE.md is loaded by every agent, so the hard rules (never write to data/, do not change a
// computation that feeds a manuscript number, external data is read-only) are not repeated here.
// ---------------------------------------------------------------------------------------

const PY = '"C:/Users/lomu-/AppData/Local/Programs/Python/Python311/python.exe"'

const AREAS = [
  { path: 'salmonid/config.py + salmonid/holdout.py + salmonid/util',
    focus: 'path resolution, the held-out exclusion logic, anything that could silently read the wrong file or silently skip a guard' },
  { path: 'salmonid/data + salmonid/train + salmonid/provenance + provenance/',
    focus: 'whether the published training set and training run can be rebuilt or at least verified from what is deposited; unseeded steps; absolute paths; the leak audit' },
  { path: 'salmonid/detect',
    focus: 'inference and tracking: coordinate conventions, thresholds, frame sampling, silent data-loss paths, whether defaults equal the published configuration' },
  { path: 'salmonid/count',
    focus: 'the counting adapter, the technician-record loaders, the direction rule and its validation; which modules are on the published path and which are legacy' },
  { path: 'salmonid/eval',
    focus: 'the held-out score, the test-set construction, the development gates (can any pass with no data or never fail?)' },
  { path: 'salmonid/results + salmonid/paper + data/',
    focus: 'statistics: estimator correctness, seeds, bootstrap sizes, day sets, denominators; does every committed table regenerate from committed code; is data/README.md accurate column by column' },
  { path: 'salmonid/figures',
    focus: 'does each figure script read only what it says it reads; do annotations printed on figures equal the numbers in the tables and captions' },
  { path: 'README.md + docs/ + CITATION.cff + LICENSE + environment/ + requirements*.txt + pyproject.toml + Makefile + run_tier1.py',
    focus: 'would an outside reader on macOS or Linux succeed by following the documents literally; stale version numbers, wrong table or figure numbers, commands that do not exist, missing citations or licences' },
  { path: 'docs/PAPER_TO_CODE.md against ../salmonid-classifier-paper/manuscript.md',
    focus: 'every number, table and figure in the CURRENT manuscript: is it mapped to a command, does the mapped command give the printed value, is anything in the manuscript missing from the map' },
]

const FINDINGS = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          title: { type: 'string', description: 'one sentence stating the defect' },
          severity: { type: 'string', enum: ['blocker', 'major', 'minor'] },
          file: { type: 'string', description: 'repo-relative path' },
          line: { type: 'integer' },
          evidence: { type: 'string', description: 'the exact text, command and output, or values that show it' },
          consequence: { type: 'string', description: 'what a reviewer or reader would get wrong because of it' },
          needs_author: { type: 'boolean', description: 'true if only a human decision can resolve it' },
        },
        required: ['title', 'severity', 'file', 'evidence', 'consequence', 'needs_author'],
      },
    },
  },
  required: ['findings'],
}

const VERDICT = {
  type: 'object',
  properties: {
    refuted: { type: 'boolean' },
    already_registered: { type: 'boolean', description: 'true if docs/REPRODUCIBILITY_REVIEW.md already records this' },
    reason: { type: 'string' },
  },
  required: ['refuted', 'already_registered', 'reason'],
}

const BASELINE = {
  type: 'object',
  properties: {
    passed: { type: 'integer' },
    failed: { type: 'integer' },
    last_lines: { type: 'string' },
  },
  required: ['passed', 'failed', 'last_lines'],
}

const selected = Array.isArray(args) && args.length
  ? AREAS.filter(a => args.some(x => a.path.includes(x)))
  : AREAS
if (selected.length < AREAS.length) {
  log(`Auditing ${selected.length} of ${AREAS.length} areas (args filter); the rest are NOT covered by this run.`)
}

phase('Baseline')
const baseline = await agent(
  `Run ${PY} run_tier1.py from the repository root and report the counts in its final RESULT line and its last 15 lines. Do not edit any file.`,
  { label: 'tier1 baseline', phase: 'Baseline', schema: BASELINE, effort: 'low' })
if (baseline && baseline.failed > 0) {
  log(`Tier 1 baseline has ${baseline.failed} failure(s). The audit continues, but fix those first.`)
}

const reviewPrompt = a => `You are auditing one area of a public code-and-data deposit for a fisheries manuscript, for reproducibility. READ-ONLY: do not edit, create or delete any file in the repository; write scratch output only to your scratchpad.

Area: ${a.path}
Look especially for: ${a.focus}

Method: read every file in the area in full. Where a claim can be tested by running something from committed data, run it (interpreter: ${PY}; set PYTHONPATH to the repository root; pass an output directory outside data/). Read docs/REPRODUCIBILITY_REVIEW.md first and do NOT report anything it already records, whether fixed or open.

Report only defects you can show with evidence: a docstring or document statement that the code contradicts; a default that does not equal the published configuration; a result that does not regenerate; a silent failure path; an unseeded random step; a path or name that does not exist; a number that differs from the manuscript at ../salmonid-classifier-paper/manuscript.md. Style preferences are not findings. If the area is clean, return an empty list.`

const verifyPrompt = (f, a) => `Try to REFUTE this audit finding about the repository. Read the cited file and anything else needed; run a command if that settles it. READ-ONLY: edit nothing.

Area: ${a.path}
Finding: ${f.title}
File: ${f.file}${f.line ? ':' + f.line : ''}
Evidence offered: ${f.evidence}
Claimed consequence: ${f.consequence}

Set refuted=true if the evidence is wrong, the behaviour is intended and documented, or you cannot reproduce it; if you are unsure, refuted=true. Set already_registered=true if docs/REPRODUCIBILITY_REVIEW.md already records the same issue.`

const perArea = await pipeline(
  selected,
  a => agent(reviewPrompt(a), { label: `review: ${a.path.split(' ')[0]}`, phase: 'Review', schema: FINDINGS }),
  (review, a) => parallel(((review && review.findings) || []).map(f => () =>
    agent(verifyPrompt(f, a), { label: `verify: ${f.file}`, phase: 'Verify', schema: VERDICT })
      .then(v => ({ ...f, area: a.path, verdict: v })))),
)

const all = perArea.filter(Boolean).flat().filter(Boolean)
const confirmed = all.filter(f => f.verdict && !f.verdict.refuted && !f.verdict.already_registered)
const dropped = all.length - confirmed.length
log(`${all.length} findings raised, ${confirmed.length} confirmed and new, ${dropped} refuted or already registered.`)

phase('Synthesize')
const order = { blocker: 0, major: 1, minor: 2 }
confirmed.sort((x, y) => order[x.severity] - order[y.severity])
const report = confirmed.length
  ? await agent(
      `Write a short register entry for each of these confirmed reproducibility findings, in the format used by docs/REPRODUCIBILITY_REVIEW.md (read it for the format and for the next free identifier). Do not edit that file: return the entries as text so the coordinating session can decide what to add. Findings (JSON): ${JSON.stringify(confirmed)}`,
      { label: 'draft register entries', phase: 'Synthesize' })
  : 'No new confirmed findings.'

return { baseline, areas_audited: selected.map(a => a.path), raised: all.length, confirmed, report }
