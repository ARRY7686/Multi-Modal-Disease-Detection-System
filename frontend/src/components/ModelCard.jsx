import { useEffect, useState } from 'react'
import { getModelInfo, errorMessage } from '../api'
import { MODALITIES, MODES, MODE_ORDER, fixed, pct, prettyLabel } from '../lib/domain'

const withCi = (v, ci) =>
  v == null ? '–' : ci && ci[0] != null ? `${fixed(v)} [${fixed(ci[0])}, ${fixed(ci[1])}]` : fixed(v)

function Legend({ modes }) {
  return (
    <div className="flex flex-wrap gap-4 text-xs text-ink-2">
      {modes.map((m) => (
        <span key={m} className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full" style={{ background: MODES[m].color }} />
          {MODES[m].label}
        </span>
      ))}
    </div>
  )
}

// Dot plot: per-class chest AUROC, image only vs multimodal. 0.5 = chance.
function PerClassDots({ results }) {
  const [hover, setHover] = useState(null)
  const modes = ['multimodal', 'image_only'].filter((m) => results[m]?.chest?.per_class_auroc)
  if (modes.length === 0) return null
  const ref = results[modes[0]].chest.per_class_auroc
  const classes = Object.keys(ref).sort((a, b) => (ref[b] ?? 0) - (ref[a] ?? 0))
  const W = 640, row = 26, left = 150, right = 24, top = 24
  const H = top + classes.length * row + 28
  const x = (v) => left + ((v - 0.5) / 0.5) * (W - left - right)
  const ticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Per-class chest AUROC">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={top - 8} y2={H - 24} stroke="#2c2c2a" />
            <text x={x(t)} y={H - 8} textAnchor="middle" fontSize="11" fill="#898781" className="num">
              {t.toFixed(1)}
            </text>
          </g>
        ))}
        <text x={x(0.5) + 4} y={top - 12} fontSize="10" fill="#898781">chance</text>
        {classes.map((c, i) => {
          const y = top + i * row + row / 2
          const vals = modes.map((m) => results[m].chest.per_class_auroc[c])
          return (
            <g key={c}>
              <text x={left - 12} y={y + 4} textAnchor="end" fontSize="12" fill="#c3c2b7">
                {prettyLabel(c)}
              </text>
              {vals.length === 2 && vals.every((v) => v != null) && (
                <line x1={x(vals[0])} x2={x(vals[1])} y1={y} y2={y} stroke="#383835" strokeWidth="2" />
              )}
              {modes.map((m, j) =>
                vals[j] == null ? null : (
                  <circle
                    key={m}
                    cx={x(vals[j])}
                    cy={y}
                    r="5"
                    fill={MODES[m].color}
                    stroke="#1a1a19"
                    strokeWidth="2"
                  />
                ),
              )}
              <rect
                x={0}
                y={y - row / 2}
                width={W}
                height={row}
                fill="transparent"
                onMouseEnter={() => setHover({ c, y })}
                onMouseLeave={() => setHover(null)}
              />
            </g>
          )
        })}
      </svg>
      {hover && (
        <div
          className="pointer-events-none absolute right-2 rounded-lg border border-line-strong bg-surface-2 px-3 py-2 text-xs shadow-xl"
          style={{ top: `${(hover.y / H) * 100}%` }}
        >
          <div className="mb-1 font-medium text-ink">{prettyLabel(hover.c)}</div>
          {modes.map((m) => (
            <div key={m} className="flex items-center justify-between gap-4 text-ink-2">
              <span className="inline-flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full" style={{ background: MODES[m].color }} />
                {MODES[m].label}
              </span>
              <span className="num">{fixed(results[m].chest.per_class_auroc[hover.c])}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// Validation AUROC per epoch, one 2px line per mode, crosshair on hover.
function TrainingCurve({ history }) {
  const [hover, setHover] = useState(null)
  if (!history?.length) return null
  const modes = MODE_ORDER.filter((m) => history.some((h) => h.val_auc?.[m] != null))
  const vals = history.flatMap((h) => modes.map((m) => h.val_auc[m])).filter((v) => v != null)
  const lo = Math.floor(Math.min(...vals) * 20) / 20
  const hi = Math.min(1, Math.ceil(Math.max(...vals) * 20) / 20)
  const W = 640, H = 240, l = 44, r = 16, t = 12, b = 32
  const x = (i) => l + (history.length === 1 ? 0 : (i / (history.length - 1)) * (W - l - r))
  const y = (v) => t + (1 - (v - lo) / (hi - lo || 1)) * (H - t - b)
  const ticks = Array.from({ length: Math.round((hi - lo) / 0.05) + 1 }, (_, i) => lo + i * 0.05)

  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        role="img"
        aria-label="Validation AUROC per epoch"
        onMouseLeave={() => setHover(null)}
        onMouseMove={(e) => {
          const box = e.currentTarget.getBoundingClientRect()
          const px = ((e.clientX - box.left) / box.width) * W
          const i = Math.round(((px - l) / (W - l - r)) * (history.length - 1))
          setHover(Math.max(0, Math.min(history.length - 1, i)))
        }}
      >
        {ticks.map((v) => (
          <g key={v}>
            <line x1={l} x2={W - r} y1={y(v)} y2={y(v)} stroke="#2c2c2a" />
            <text x={l - 8} y={y(v) + 4} textAnchor="end" fontSize="11" fill="#898781">
              {v.toFixed(2)}
            </text>
          </g>
        ))}
        {history.map((h, i) => (
          <text key={i} x={x(i)} y={H - 10} textAnchor="middle" fontSize="11" fill="#898781">
            {h.epoch}
          </text>
        ))}
        {hover != null && <line x1={x(hover)} x2={x(hover)} y1={t} y2={H - b} stroke="#383835" />}
        {modes.map((m) => (
          <g key={m}>
            <polyline
              fill="none"
              stroke={MODES[m].color}
              strokeWidth="2"
              strokeLinejoin="round"
              points={history.map((h, i) => `${x(i)},${y(h.val_auc[m])}`).join(' ')}
            />
            {hover != null && (
              <circle
                cx={x(hover)}
                cy={y(history[hover].val_auc[m])}
                r="4.5"
                fill={MODES[m].color}
                stroke="#1a1a19"
                strokeWidth="2"
              />
            )}
          </g>
        ))}
      </svg>
      {hover != null && (
        <div className="pointer-events-none absolute left-14 top-2 rounded-lg border border-line-strong bg-surface-2 px-3 py-2 text-xs shadow-xl">
          <div className="mb-1 font-medium text-ink">
            Epoch {history[hover].epoch}
            {history[hover].frozen && <span className="text-ink-muted"> (backbones frozen)</span>}
          </div>
          {modes.map((m) => (
            <div key={m} className="flex items-center justify-between gap-4 text-ink-2">
              <span className="inline-flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full" style={{ background: MODES[m].color }} />
                {MODES[m].label}
              </span>
              <span className="num">{fixed(history[hover].val_auc[m], 4)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function Finding({ results }) {
  const mm = results.multimodal?.chest, io = results.image_only?.chest
  if (!mm?.macro_auroc_ci || !io?.macro_auroc_ci) return null
  const overlap = mm.macro_auroc_ci[0] <= io.macro_auroc_ci[1] && io.macro_auroc_ci[0] <= mm.macro_auroc_ci[1]
  const diff = mm.macro_auroc - io.macro_auroc
  return (
    <p className="text-sm leading-relaxed text-ink-2">
      On chest X-rays, fusing the patient details changes macro AUROC by{' '}
      <span className="num text-ink">{diff >= 0 ? '+' : ''}{diff.toFixed(3)}</span> versus the image alone.{' '}
      {overlap
        ? 'The 95% confidence intervals overlap, so the metadata adds no measurable accuracy: the image does the work.'
        : 'The 95% confidence intervals do not overlap, so the difference is measurable.'}
    </p>
  )
}

export default function ModelCard() {
  const [info, setInfo] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    getModelInfo().then(setInfo).catch((e) => setError(errorMessage(e)))
  }, [])

  if (error) return <div className="card card-pad text-sm text-status-serious">{error}</div>
  if (!info) return <div className="card card-pad text-sm text-ink-muted">Loading model card…</div>
  if (!info.weights_loaded)
    return (
      <div className="card card-pad text-sm text-ink-2">
        No trained checkpoint is loaded, so there are no results to show. Download the weights into{' '}
        <code className="text-ink">models/</code> and restart the API.
      </div>
    )

  const R = info.test_results
  const modes = MODE_ORDER.filter((m) => R[m])
  const counts = info.data_counts || {}

  return (
    <div className="space-y-6">
      <section className="card card-pad">
        <div className="eyebrow">Loaded checkpoint</div>
        <div className="mt-3 grid grid-cols-2 gap-4 sm:grid-cols-4">
          <div>
            <div className="text-xs text-ink-muted">Image encoder</div>
            <div className="text-sm text-ink">{info.image_backbone}</div>
          </div>
          <div>
            <div className="text-xs text-ink-muted">Text encoder</div>
            <div className="truncate text-sm text-ink" title={info.text_model_name}>{info.text_model_name}</div>
          </div>
          <div>
            <div className="text-xs text-ink-muted">Chest training patients</div>
            <div className="num text-sm text-ink">{pct(info.chest_train_fraction, 0)}</div>
          </div>
          <div>
            <div className="text-xs text-ink-muted">Modality router accuracy</div>
            <div className="num text-sm text-ink">{pct(info.modality_classifier_test_accuracy, 2)}</div>
          </div>
        </div>
        <div className="mt-5 border-t border-line pt-4">
          <Finding results={R} />
        </div>
      </section>

      <section className="card card-pad">
        <h2 className="card-title">Held-out test results</h2>
        <p className="mt-1 text-xs text-ink-muted">Scored once, after all tuning. 95% bootstrap intervals in brackets.</p>
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[640px] text-xs">
            <thead>
              <tr className="border-b border-line text-left text-ink-muted">
                <th className="py-2 pr-4 font-medium">Mode</th>
                <th className="py-2 pr-4 font-medium">Chest macro AUROC</th>
                <th className="py-2 pr-4 font-medium">Skin balanced acc.</th>
                <th className="py-2 pr-4 font-medium">Skin AUROC</th>
                <th className="py-2 pr-4 font-medium">Brain acc.</th>
                <th className="py-2 pr-4 font-medium">Brain AUROC</th>
                <th className="py-2 font-medium">ECE</th>
              </tr>
            </thead>
            <tbody className="num">
              {modes.map((m) => (
                <tr key={m} className="border-b border-line/60 text-ink-2">
                  <td className="py-2.5 pr-4 text-ink">
                    <span className="inline-flex items-center gap-1.5">
                      <span className="h-2 w-2 rounded-full" style={{ background: MODES[m].color }} />
                      {MODES[m].label}
                    </span>
                  </td>
                  <td className="py-2.5 pr-4">{withCi(R[m].chest?.macro_auroc, R[m].chest?.macro_auroc_ci)}</td>
                  <td className="py-2.5 pr-4">{withCi(R[m].skin?.balanced_accuracy, R[m].skin?.balanced_accuracy_ci)}</td>
                  <td className="py-2.5 pr-4">{fixed(R[m].skin?.macro_auroc)}</td>
                  <td className="py-2.5 pr-4">{fixed(R[m].brain?.accuracy)}</td>
                  <td className="py-2.5 pr-4">{fixed(R[m].brain?.macro_auroc)}</td>
                  <td className="py-2.5">{fixed(R[m].ece)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <div className="grid items-start gap-6 xl:grid-cols-2">
        <section className="card card-pad">
          <h2 className="card-title">Chest X-ray: AUROC per finding</h2>
          <div className="mb-3 mt-2">
            <Legend modes={['multimodal', 'image_only'].filter((m) => R[m])} />
          </div>
          <PerClassDots results={R} />
        </section>
        <section className="card card-pad">
          <h2 className="card-title">Validation AUROC during training</h2>
          <p className="mt-1 text-xs text-ink-muted">All 25 classes pooled, per epoch. The best epoch was kept.</p>
          <div className="mb-3 mt-2">
            <Legend modes={modes} />
          </div>
          <TrainingCurve history={info.training_history} />
        </section>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <section className="card card-pad">
          <h2 className="card-title">Data</h2>
          <table className="mt-3 w-full text-xs">
            <thead>
              <tr className="border-b border-line text-left text-ink-muted">
                <th className="py-2 font-medium">Split</th>
                {Object.keys(MODALITIES).map((k) => (
                  <th key={k} className="py-2 text-right font-medium">{MODALITIES[k].short}</th>
                ))}
              </tr>
            </thead>
            <tbody className="num">
              {['train', 'val', 'test'].map((s) => (
                <tr key={s} className="border-b border-line/60 text-ink-2">
                  <td className="py-2 capitalize text-ink">{s}</td>
                  {Object.keys(MODALITIES).map((k) => (
                    <td key={k} className="py-2 text-right">{counts[s]?.[k]?.toLocaleString() ?? '–'}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </section>
        <section className="card card-pad">
          <h2 className="card-title">Evaluation protocol</h2>
          <dl className="mt-3 space-y-3 text-xs">
            {Object.entries(info.evaluation_protocol || {}).map(([k, v]) => (
              <div key={k}>
                <dt className="eyebrow">{k}</dt>
                <dd className="mt-0.5 leading-relaxed text-ink-2">{v}</dd>
              </div>
            ))}
          </dl>
        </section>
      </div>
    </div>
  )
}
