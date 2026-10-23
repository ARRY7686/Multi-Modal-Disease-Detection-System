import { MODALITIES, MODES, SOURCE_LABELS, pct, prettyLabel } from '../lib/domain'
import CompareView from './CompareView'
import HeatmapViewer from './HeatmapViewer'
import { Alert, Clock, Download, Printer } from './Icons'
import PredictionList from './PredictionList'
import TextExplanation from './TextExplanation'

function Stat({ label, children }) {
  return (
    <div>
      <div className="eyebrow">{label}</div>
      <div className="mt-1 text-sm text-ink">{children}</div>
    </div>
  )
}

function downloadJson(entry) {
  const blob = new Blob([JSON.stringify({ input: entry.input, results: entry.results }, null, 2)], {
    type: 'application/json',
  })
  const url = URL.createObjectURL(blob)
  const a = Object.assign(document.createElement('a'), { href: url, download: `analysis-${entry.id}.json` })
  a.click()
  URL.revokeObjectURL(url)
}

export default function ResultsPanel({ entry, original }) {
  const main = entry.results[entry.primary]
  const top = main.predictions[0]
  const flagged = main.predictions.filter((p) => p.positive)

  return (
    <div className="space-y-6">
      <section className="card card-pad print-ink">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="eyebrow">Top finding</div>
            <div className="mt-1 text-2xl font-semibold text-ink">{prettyLabel(top.disease)}</div>
            <div className="mt-0.5 text-sm text-ink-2">
              {pct(top.confidence)} confidence ·{' '}
              {top.positive ? 'above its decision threshold' : 'below its decision threshold'}
            </div>
          </div>
          <div className="flex gap-2 no-print">
            <button type="button" onClick={() => downloadJson(entry)} className="btn-ghost px-3 py-1.5 text-xs">
              <Download className="h-3.5 w-3.5" /> JSON
            </button>
            <button type="button" onClick={() => window.print()} className="btn-ghost px-3 py-1.5 text-xs">
              <Printer className="h-3.5 w-3.5" /> Report
            </button>
          </div>
        </div>
        <div className="mt-5 grid grid-cols-2 gap-4 border-t border-line pt-4 sm:grid-cols-4">
          <Stat label="Modality">
            {MODALITIES[main.modality]?.label}
            <div className="text-xs text-ink-muted">
              {SOURCE_LABELS[main.modality_source]}
              {main.modality_confidence != null && `, ${pct(main.modality_confidence, 0)}`}
            </div>
          </Stat>
          <Stat label="Mode">
            <span className="inline-flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full" style={{ background: MODES[main.mode].color }} />
              {MODES[main.mode].label}
            </span>
          </Stat>
          <Stat label="Flagged">
            <span className="num">{flagged.length}</span> of {main.predictions.length} shown
          </Stat>
          <Stat label="Latency">
            <span className="inline-flex items-center gap-1.5 num">
              <Clock className="h-3.5 w-3.5 text-ink-muted" />
              {Math.round(main.inference_time_ms)} ms
            </span>
          </Stat>
        </div>
      </section>

      {main.warnings?.length > 0 && (
        <div className="flex gap-3 rounded-2xl border border-status-warning/40 bg-status-warning/5 p-4 text-sm text-ink-2">
          <Alert className="mt-0.5 h-4 w-4 shrink-0 text-status-warning" />
          <div className="space-y-1">
            {main.warnings.map((w) => (
              <p key={w}>{w}</p>
            ))}
          </div>
        </div>
      )}

      <section className="card card-pad print-ink">
        <h2 className="card-title mb-4">Predictions</h2>
        <PredictionList result={main} />
      </section>

      <CompareView results={entry.results} />
      <HeatmapViewer explanation={main.explanation} original={original} />
      <TextExplanation explanation={main.explanation} />
    </div>
  )
}
