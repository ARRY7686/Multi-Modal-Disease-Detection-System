import { MODES, pct, prettyLabel } from '../lib/domain'
import { Alert } from './Icons'

// One horizontal bar per class: confidence from 0, with the class's
// validation-fitted decision threshold as a tick.
export default function PredictionList({ result }) {
  const color = MODES[result.mode]?.color || '#9085e9'
  return (
    <div>
      <ul className="space-y-4">
        {result.predictions.map((p) => (
          <li key={p.label_index} className="group relative">
            <div className="mb-1.5 flex items-baseline justify-between gap-3">
              <span className="truncate text-[15px] text-ink">{prettyLabel(p.disease)}</span>
              <span className="flex shrink-0 items-center gap-3">
                {p.positive && (
                  <span className="inline-flex items-center gap-1 text-[11px] uppercase tracking-[0.1em] text-status-serious">
                    <Alert className="h-3 w-3" />
                    Above threshold
                  </span>
                )}
                <span className="num w-14 text-right text-[15px] text-ink">{pct(p.confidence)}</span>
              </span>
            </div>
            <div className="relative h-1.5 rounded-full bg-white/[0.07]">
              <div
                className="h-1.5 rounded-full transition-[width] duration-500"
                style={{ width: `${Math.max(p.confidence * 100, 0.8)}%`, background: color }}
              />
              <div className="absolute -top-1 h-3.5 w-px bg-ink-2" style={{ left: `${p.threshold * 100}%` }} />
            </div>
            <div className="pointer-events-none absolute right-0 top-full z-10 mt-2 hidden w-56 rounded-xl border border-line-strong bg-surface-2 p-3 text-xs shadow-2xl group-hover:block">
              <div className="mb-1 text-ink">{prettyLabel(p.disease)}</div>
              <div className="flex justify-between text-ink-2">
                <span>Confidence</span>
                <span className="num">{pct(p.confidence)}</span>
              </div>
              <div className="flex justify-between text-ink-2">
                <span>Decision threshold</span>
                <span className="num">{pct(p.threshold)}</span>
              </div>
              <div className="mt-1 text-ink-muted">{p.positive ? 'At or above threshold: flagged.' : 'Below threshold: not flagged.'}</div>
            </div>
          </li>
        ))}
      </ul>
      <p className="mt-5 text-xs leading-relaxed text-ink-muted">
        {result.calibrated ? 'Temperature-scaled on validation data.' : 'Raw model outputs.'} The tick is each class&apos;s
        threshold, fitted on validation to maximise F1. Top {result.predictions.length} classes for this modality.
      </p>
    </div>
  )
}
