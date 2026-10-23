import { useState } from 'react'
import { MODALITIES, MODES, SOURCE_LABELS, detailsOverride, pct, prettyLabel } from '../lib/domain'
import CompareView, { compareModes } from './CompareView'
import { Alert, Download, Printer } from './Icons'
import PredictionList from './PredictionList'
import Section from './Section'
import TextExplanation from './TextExplanation'

function Stat({ label, children }) {
  return (
    <div className="border-t border-line-strong pt-3">
      <div className="text-[11px] uppercase tracking-[0.12em] text-ink-muted">{label}</div>
      <div className="mt-1.5 text-[15px] text-ink">{children}</div>
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

const two = (n) => String(n).padStart(2, '0')

export default function ResultsPanel({ entry }) {
  const [asTable, setAsTable] = useState(false)
  const main = entry.results[entry.primary]
  const flagged = main.predictions.filter((p) => p.positive)
  const override = detailsOverride(entry.results)
  const hasCompare = compareModes(entry.results).length > 1
  const hasText = main.explanation?.highlighted_text || main.explanation?.important_tokens?.length
  let n = 0

  return (
    <div className="space-y-6">
      <Section
        id="findings"
        n={two(++n)}
        highlight="Findings"
        title="for this case"
        aside={
          <>
            <button type="button" onClick={() => downloadJson(entry)} className="pill pill-sm no-print">
              <Download className="h-3 w-3" /> JSON
            </button>
            <button type="button" onClick={() => window.print()} className="pill pill-sm no-print">
              <Printer className="h-3 w-3" /> Report
            </button>
          </>
        }
      >
        {override && (
          <div className="mb-8 flex gap-4 rounded-2xl border border-status-warning/40 bg-status-warning/[0.07] p-5">
            <Alert className="mt-0.5 h-5 w-5 shrink-0 text-status-warning" />
            <div className="text-sm leading-relaxed text-ink-2">
              <p className="text-[15px] text-ink">The patient details changed the top finding.</p>
              <p className="mt-1.5">
                Image only: <span className="text-ink">{prettyLabel(override.image.disease)}</span>{' '}
                <span className="num">{pct(override.image.confidence)}</span>
                {override.image.positive && ' (above threshold)'}. With details:{' '}
                <span className="text-ink">{prettyLabel(override.fused.disease)}</span>{' '}
                <span className="num">{pct(override.fused.confidence)}</span>.
              </p>
              <p className="mt-1.5 text-ink-muted">
                On held-out data the image alone scores as well as the fused model, and the fused model leans on
                age and site patterns from its training set (for example, young age pushes towards benign nevi). Do not
                let the details overrule the image.
              </p>
            </div>
          </div>
        )}
        <div className="grid gap-10 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
          <div className="grid grid-cols-2 content-start gap-x-6 gap-y-5">
            <Stat label="Modality">
              {MODALITIES[main.modality]?.label}
              <div className="text-xs text-ink-muted">
                {SOURCE_LABELS[main.modality_source]}
                {main.modality_confidence != null && `, ${pct(main.modality_confidence, 0)}`}
              </div>
            </Stat>
            <Stat label="Mode">
              <span className="inline-flex items-center gap-2">
                <span className="h-2 w-2 rounded-full" style={{ background: MODES[main.mode].color }} />
                {MODES[main.mode].label}
              </span>
            </Stat>
            <Stat label="Flagged">
              <span className="num">{flagged.length}</span>
              <span className="text-ink-muted"> of {main.predictions.length} shown</span>
            </Stat>
            <Stat label="Latency">
              <span className="num">{Math.round(main.inference_time_ms)}</span>
              <span className="text-ink-muted"> ms</span>
            </Stat>
            {main.warnings?.length > 0 && (
              <div className="col-span-2 flex gap-3 rounded-2xl border border-status-warning/30 bg-status-warning/5 p-4 text-sm text-ink-2">
                <Alert className="mt-0.5 h-4 w-4 shrink-0 text-status-warning" />
                <div className="space-y-1">
                  {main.warnings.map((w) => (
                    <p key={w}>{w}</p>
                  ))}
                </div>
              </div>
            )}
          </div>
          <PredictionList result={main} />
        </div>
      </Section>

      {hasCompare && (
        <Section
          id="compare"
          n={two(++n)}
          highlight="Compare"
          title="image, details and fusion"
          aside={
            <button type="button" onClick={() => setAsTable((v) => !v)} className="pill pill-sm no-print">
              {asTable ? 'Chart' : 'Table'}
            </button>
          }
        >
          <CompareView results={entry.results} asTable={asTable} />
        </Section>
      )}

      {hasText && (
        <Section id="attention" n={two(++n)} highlight="Attention" title="over the patient details">
          <TextExplanation explanation={main.explanation} />
        </Section>
      )}
    </div>
  )
}
