import { useState } from 'react'
import { MODES, MODE_ORDER, pct, prettyLabel } from '../lib/domain'

function Legend({ modes }) {
  return (
    <div className="flex flex-wrap gap-4 text-xs text-ink-2">
      {modes.map((m) => (
        <span key={m} className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ background: MODES[m].color }} />
          {MODES[m].label}
        </span>
      ))}
    </div>
  )
}

// Same case scored three ways: image only, details only, and fused.
export default function CompareView({ results }) {
  const [asTable, setAsTable] = useState(false)
  const modes = MODE_ORDER.filter((m) => results[m])
  if (modes.length < 2) return null

  // Rows: every class any mode ranked, ordered by the multimodal score
  const score = (mode, idx) => results[mode]?.predictions.find((p) => p.label_index === idx)
  const rows = new Map()
  modes.forEach((m) => results[m].predictions.forEach((p) => rows.set(p.label_index, p.disease)))
  const ordered = [...rows.entries()].sort(
    ([a], [b]) => (score(modes[0], b)?.confidence ?? -1) - (score(modes[0], a)?.confidence ?? -1),
  )

  const top = Object.fromEntries(modes.map((m) => [m, results[m].predictions[0]]))
  const agree = modes.every((m) => top[m]?.label_index === top[modes[0]]?.label_index)

  return (
    <section className="card card-pad">
      <div className="mb-3 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="card-title">Mode comparison</h2>
          <p className="mt-0.5 text-xs text-ink-muted">
            {agree
              ? `All modes agree on ${prettyLabel(top[modes[0]].disease)} as the top class.`
              : 'The modes disagree on the top class.'}
          </p>
        </div>
        <button type="button" onClick={() => setAsTable((v) => !v)} className="btn-ghost px-3 py-1.5 text-xs no-print">
          {asTable ? 'Show chart' : 'Show table'}
        </button>
      </div>

      <div className="mb-4">
        <Legend modes={modes} />
      </div>

      {asTable ? (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-line text-left text-ink-muted">
                <th className="py-2 pr-3 font-medium">Class</th>
                {modes.map((m) => (
                  <th key={m} className="py-2 pr-3 text-right font-medium">
                    {MODES[m].label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ordered.map(([idx, name]) => (
                <tr key={idx} className="border-b border-line/60">
                  <td className="py-2 pr-3 text-ink">{prettyLabel(name)}</td>
                  {modes.map((m) => (
                    <td key={m} className="num py-2 pr-3 text-right text-ink-2">
                      {score(m, idx) ? pct(score(m, idx).confidence) : 'not in top 5'}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <ul className="space-y-4">
          {ordered.map(([idx, name]) => (
            <li key={idx}>
              <div className="mb-1.5 text-sm text-ink">{prettyLabel(name)}</div>
              <div className="space-y-[2px]">
                {modes.map((m) => {
                  const s = score(m, idx)
                  return (
                    <div key={m} className="group relative grid grid-cols-[1fr_3.5rem] items-center gap-3">
                      <div className="h-1.5 rounded-full bg-surface-3">
                        {s && (
                          <div
                            className="h-1.5 rounded-full"
                            style={{ width: `${Math.max(s.confidence * 100, 0.8)}%`, background: MODES[m].color }}
                          />
                        )}
                      </div>
                      <span className="num text-right text-[11px] text-ink-2">{s ? pct(s.confidence) : '–'}</span>
                      <div className="pointer-events-none absolute -top-8 left-0 z-10 hidden rounded-md border border-line-strong bg-surface-2 px-2 py-1 text-[11px] text-ink shadow-lg group-hover:block">
                        {MODES[m].label}: {s ? pct(s.confidence) : 'not in top 5'}
                      </div>
                    </div>
                  )
                })}
              </div>
            </li>
          ))}
        </ul>
      )}
      <p className="mt-4 text-[11px] leading-relaxed text-ink-muted">
        On the held-out test sets, image-only and multimodal scores sit within each other&apos;s confidence intervals:
        the recorded metadata adds little on top of the image. See the Model card tab.
      </p>
    </section>
  )
}
