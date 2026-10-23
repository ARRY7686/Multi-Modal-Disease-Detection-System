import { MODES, pct, prettyLabel } from '../lib/domain'
import { Alert } from './Icons'

// One horizontal bar per class: confidence from 0, with the class's
// validation-fitted decision threshold as a tick.
export default function PredictionList({ result }) {
  const color = MODES[result.mode]?.color || '#3987e5'
  return (
    <div>
      <ul className="space-y-3">
        {result.predictions.map((p) => (
          <li key={p.label_index} className="group relative">
            <div className="mb-1 flex items-baseline justify-between gap-3">
              <span className="truncate text-sm text-ink">{prettyLabel(p.disease)}</span>
              <span className="flex shrink-0 items-center gap-2">
                {p.positive && (
                  <span className="inline-flex items-center gap-1 rounded-md border border-status-serious/40 px-1.5 py-0.5 text-[11px] font-medium text-status-serious">
                    <Alert className="h-3 w-3" />
                    Above threshold
                  </span>
                )}
                <span className="num w-14 text-right text-sm font-medium text-ink">{pct(p.confidence)}</span>
              </span>
            </div>
            <div className="relative h-2 rounded-full bg-surface-3">
              <div
                className="h-2 rounded-full transition-[width] duration-500"
                style={{ width: `${Math.max(p.confidence * 100, 0.8)}%`, background: color }}
              />
              <div
                className="absolute -top-1 h-4 w-0.5 rounded bg-ink-2"
                style={{ left: `calc(${p.threshold * 100}% - 1px)` }}
              />
            </div>
            <div className="pointer-events-none absolute right-0 top-full z-10 mt-2 hidden w-56 rounded-lg border border-line-strong bg-surface-2 p-2.5 text-xs shadow-xl group-hover:block">
              <div className="mb-1 font-medium text-ink">{prettyLabel(p.disease)}</div>
              <div className="flex justify-between text-ink-2">
                <span>Confidence</span>
                <span className="num">{pct(p.confidence)}</span>
              </div>
              <div className="flex justify-between text-ink-2">
                <span>Decision threshold</span>
                <span className="num">{pct(p.threshold)}</span>
              </div>
              <div className="mt-1 text-ink-muted">
                {p.positive ? 'At or above threshold: flagged.' : 'Below threshold: not flagged.'}
              </div>
            </div>
          </li>
        ))}
      </ul>
      <p className="mt-4 text-[11px] leading-relaxed text-ink-muted">
        {result.calibrated ? 'Scores are temperature-scaled on validation data.' : 'Scores are raw model outputs.'}{' '}
        The tick on each bar is that class&apos;s threshold, fitted on validation to maximise F1. Showing the top{' '}
        {result.predictions.length} classes for this modality.
      </p>
    </div>
  )
}
