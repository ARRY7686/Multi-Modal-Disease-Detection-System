import { useState } from 'react'
import { MODES, MODE_ORDER, pct, prettyLabel } from '../lib/domain'

export function ModeLegend({ modes }) {
  return (
    <div className="flex flex-wrap gap-5 text-xs text-ink-2">
      {modes.map((m) => (
        <span key={m} className="inline-flex items-center gap-2">
          <span className="h-2 w-2 rounded-full" style={{ background: MODES[m].color }} />
          {MODES[m].label}
        </span>
      ))}
    </div>
  )
}

export const compareModes = (results) => MODE_ORDER.filter((m) => results[m])

// The same case scored three ways: image only, details only and fused.
export default function CompareView({ results, asTable }) {
  const modes = compareModes(results)
  const score = (mode, idx) => results[mode]?.predictions.find((p) => p.label_index === idx)
  const rows = new Map()
  modes.forEach((m) => results[m].predictions.forEach((p) => rows.set(p.label_index, p.disease)))
  const ordered = [...rows.entries()].sort(
    ([a], [b]) => (score(modes[0], b)?.confidence ?? -1) - (score(modes[0], a)?.confidence ?? -1),
  )
  const top = Object.fromEntries(modes.map((m) => [m, results[m].predictions[0]]))
  const agree = modes.every((m) => top[m]?.label_index === top[modes[0]]?.label_index)

  return (
    <div>
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <ModeLegend modes={modes} />
        <p className="text-xs text-ink-muted">
          {agree
            ? `All modes agree on ${prettyLabel(top[modes[0]].disease)} as the top class.`
            : 'The modes disagree on the top class.'}
        </p>
      </div>

      {asTable ? (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-line text-left text-[11px] uppercase tracking-[0.12em] text-ink-muted">
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
                <tr key={idx} className="border-b border-line">
                  <td className="py-2.5 pr-3 text-ink">{prettyLabel(name)}</td>
                  {modes.map((m) => (
                    <td key={m} className="num py-2.5 pr-3 text-right text-ink-2">
                      {score(m, idx) ? pct(score(m, idx).confidence) : 'not in top 5'}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <ul className="grid gap-x-10 gap-y-6 md:grid-cols-2">
          {ordered.map(([idx, name]) => (
            <li key={idx}>
              <div className="mb-2 text-sm text-ink">{prettyLabel(name)}</div>
              <div className="space-y-[3px]">
                {modes.map((m) => {
                  const s = score(m, idx)
                  return (
                    <div key={m} className="group relative grid grid-cols-[1fr_3.5rem] items-center gap-3">
                      <div className="h-1.5 rounded-full bg-white/[0.07]">
                        {s && (
                          <div
                            className="h-1.5 rounded-full"
                            style={{ width: `${Math.max(s.confidence * 100, 0.8)}%`, background: MODES[m].color }}
                          />
                        )}
                      </div>
                      <span className="num text-right text-[11px] text-ink-2">{s ? pct(s.confidence) : '–'}</span>
                      <div className="pointer-events-none absolute -top-8 left-0 z-10 hidden rounded-lg border border-line-strong bg-surface-2 px-2 py-1 text-[11px] text-ink shadow-lg group-hover:block">
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
      <p className="mt-6 text-xs leading-relaxed text-ink-muted">
        On the held-out test sets, image-only and multimodal scores sit within each other&apos;s confidence intervals: the
        recorded metadata adds little on top of the image. See the Model card.
      </p>
    </div>
  )
}
