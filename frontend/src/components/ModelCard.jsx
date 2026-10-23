import { useEffect, useState } from 'react'
import { errorMessage, getModelInfo } from '../api'
import { MODALITIES, MODES, MODE_ORDER, fixed, pct, prettyLabel } from '../lib/domain'
import { ModeLegend } from './CompareView'
import Section from './Section'

// Chart chrome on the dark glass surface
const GRID = '#2a2733'
const AXIS = '#7f7a92'
const LABEL = '#bdb8cc'
const RING = '#16141d'

const withCi = (v, ci) =>
  v == null ? '–' : ci && ci[0] != null ? `${fixed(v)} [${fixed(ci[0])}, ${fixed(ci[1])}]` : fixed(v)

function Tooltip({ title, rows, style }) {
  return (
    <div
      className="pointer-events-none absolute z-10 rounded-xl border border-line-strong bg-surface-2 px-3 py-2 text-xs shadow-2xl"
      style={style}
    >
      <div className="mb-1 text-ink">{title}</div>
      {rows.map(([m, v]) => (
        <div key={m} className="flex items-center justify-between gap-5 text-ink-2">
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full" style={{ background: MODES[m].color }} />
            {MODES[m].label}
          </span>
          <span className="num">{v}</span>
        </div>
      ))}
    </div>
  )
}

// Dot plot: per-class chest AUROC, multimodal vs image only. 0.5 = chance.
function PerClassDots({ results }) {
  const [hover, setHover] = useState(null)
  const modes = ['multimodal', 'image_only'].filter((m) => results[m]?.chest?.per_class_auroc)
  if (modes.length === 0) return null
  const ref = results[modes[0]].chest.per_class_auroc
  const classes = Object.keys(ref).sort((a, b) => (ref[b] ?? 0) - (ref[a] ?? 0))
  const W = 640
  const row = 28
  const left = 150
  const right = 24
  const top = 26
  const H = top + classes.length * row + 30
  const x = (v) => left + ((v - 0.5) / 0.5) * (W - left - right)
  const ticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Per-class chest AUROC">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={top - 8} y2={H - 26} stroke={GRID} />
            <text x={x(t)} y={H - 8} textAnchor="middle" fontSize="11" fill={AXIS}>
              {t.toFixed(1)}
            </text>
          </g>
        ))}
        <text x={x(0.5) + 5} y={top - 12} fontSize="10" fill={AXIS}>
          chance
        </text>
        {classes.map((c, i) => {
          const y = top + i * row + row / 2
          const vals = modes.map((m) => results[m].chest.per_class_auroc[c])
          return (
            <g key={c}>
              <text x={left - 14} y={y + 4} textAnchor="end" fontSize="12" fill={LABEL}>
                {prettyLabel(c)}
              </text>
              {vals.length === 2 && vals.every((v) => v != null) && (
                <line x1={x(vals[0])} x2={x(vals[1])} y1={y} y2={y} stroke="#3a3645" strokeWidth="2" />
              )}
              {modes.map((m, j) =>
                vals[j] == null ? null : (
                  <circle key={m} cx={x(vals[j])} cy={y} r="5" fill={MODES[m].color} stroke={RING} strokeWidth="2" />
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
        <Tooltip
          title={prettyLabel(hover.c)}
          rows={modes.map((m) => [m, fixed(results[m].chest.per_class_auroc[hover.c])])}
          style={{ right: 8, top: `${(hover.y / H) * 100}%` }}
        />
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
  const W = 640
  const H = 260
  const l = 44
  const r = 16
  const t = 12
  const b = 32
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
            <line x1={l} x2={W - r} y1={y(v)} y2={y(v)} stroke={GRID} />
            <text x={l - 8} y={y(v) + 4} textAnchor="end" fontSize="11" fill={AXIS}>
              {v.toFixed(2)}
            </text>
          </g>
        ))}
        {history.map((h, i) => (
          <text key={i} x={x(i)} y={H - 10} textAnchor="middle" fontSize="11" fill={AXIS}>
            {h.epoch}
          </text>
        ))}
        {hover != null && <line x1={x(hover)} x2={x(hover)} y1={t} y2={H - b} stroke="#3a3645" />}
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
              <circle cx={x(hover)} cy={y(history[hover].val_auc[m])} r="4.5" fill={MODES[m].color} stroke={RING} strokeWidth="2" />
            )}
          </g>
        ))}
      </svg>
      {hover != null && (
        <Tooltip
          title={`Epoch ${history[hover].epoch}${history[hover].frozen ? ' (backbones frozen)' : ''}`}
          rows={modes.map((m) => [m, fixed(history[hover].val_auc[m], 4)])}
          style={{ left: 56, top: 8 }}
        />
      )}
    </div>
  )
}

function Finding({ results }) {
  const mm = results.multimodal?.chest
  const io = results.image_only?.chest
  if (!mm?.macro_auroc_ci || !io?.macro_auroc_ci) return null
  const overlap = mm.macro_auroc_ci[0] <= io.macro_auroc_ci[1] && io.macro_auroc_ci[0] <= mm.macro_auroc_ci[1]
  const diff = mm.macro_auroc - io.macro_auroc
  return (
    <p className="max-w-2xl text-sm leading-relaxed text-ink-2">
      On chest X-rays, fusing the patient details changes macro AUROC by{' '}
      <span className="num text-ink">
        {diff >= 0 ? '+' : ''}
        {diff.toFixed(3)}
      </span>{' '}
      against the image alone.{' '}
      {overlap
        ? 'The 95% confidence intervals overlap, so the metadata adds no measurable accuracy: the image does the work.'
        : 'The 95% confidence intervals do not overlap, so the difference is measurable.'}
    </p>
  )
}

function Hero({ label, value, sub }) {
  return (
    <div className="border-t border-line-strong pt-4">
      <div className="text-[11px] uppercase tracking-[0.14em] text-ink-muted">{label}</div>
      <div className="display num mt-2 text-5xl">{value}</div>
      {sub && <div className="mt-1 text-xs text-ink-muted">{sub}</div>}
    </div>
  )
}

export default function ModelCard() {
  const [info, setInfo] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    getModelInfo()
      .then(setInfo)
      .catch((e) => setError(errorMessage(e)))
  }, [])

  const shell = (body) => <div className="pt-10 lg:pt-16">{body}</div>
  if (error) return shell(<div className="glass p-6 text-sm text-status-serious">{error}</div>)
  if (!info) return shell(<div className="glass p-6 text-sm text-ink-muted">Loading the model card</div>)
  if (!info.weights_loaded)
    return shell(
      <div className="glass p-6 text-sm text-ink-2">
        No trained checkpoint is loaded, so there are no results to show. Put the weights in{' '}
        <code className="text-ink">models/</code> and restart the API.
      </div>,
    )

  const R = info.test_results
  const modes = MODE_ORDER.filter((m) => R[m])
  const counts = info.data_counts || {}
  const mm = R.multimodal || {}

  return shell(
    <div className="space-y-6">
      <div className="pb-6">
        <div className="eyebrow mb-4">Loaded checkpoint · held-out test sets</div>
        <h1 className="display max-w-3xl text-[40px] leading-[1.08] sm:text-[52px]">
          How well it reads, <span className="text-violet">measured once</span>
        </h1>
        <div className="mt-10 grid grid-cols-2 gap-x-8 gap-y-8 lg:grid-cols-4">
          <Hero label="Chest macro AUROC" value={fixed(mm.chest?.macro_auroc)} sub={`${counts.test?.chest?.toLocaleString() ?? '–'} X-rays`} />
          <Hero label="Skin macro AUROC" value={fixed(mm.skin?.macro_auroc)} sub={`${counts.test?.skin?.toLocaleString() ?? '–'} lesions`} />
          <Hero label="Brain accuracy" value={fixed(mm.brain?.accuracy)} sub={`${counts.test?.brain?.toLocaleString() ?? '–'} MRIs`} />
          <Hero label="Calibration error" value={fixed(mm.ece)} sub="ECE, multimodal" />
        </div>
        <div className="mt-8">
          <Finding results={R} />
        </div>
      </div>

      <Section n="01" highlight="Results" title="by mode">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[680px] text-sm">
            <thead>
              <tr className="border-b border-line text-left text-[11px] uppercase tracking-[0.12em] text-ink-muted">
                <th className="py-2.5 pr-4 font-medium">Mode</th>
                <th className="py-2.5 pr-4 font-medium">Chest macro AUROC</th>
                <th className="py-2.5 pr-4 font-medium">Skin balanced acc.</th>
                <th className="py-2.5 pr-4 font-medium">Skin AUROC</th>
                <th className="py-2.5 pr-4 font-medium">Brain acc.</th>
                <th className="py-2.5 pr-4 font-medium">Brain AUROC</th>
                <th className="py-2.5 font-medium">ECE</th>
              </tr>
            </thead>
            <tbody className="num">
              {modes.map((m) => (
                <tr key={m} className="border-b border-line text-ink-2">
                  <td className="py-3 pr-4 text-ink">
                    <span className="inline-flex items-center gap-2">
                      <span className="h-2 w-2 rounded-full" style={{ background: MODES[m].color }} />
                      {MODES[m].label}
                    </span>
                  </td>
                  <td className="py-3 pr-4">{withCi(R[m].chest?.macro_auroc, R[m].chest?.macro_auroc_ci)}</td>
                  <td className="py-3 pr-4">{withCi(R[m].skin?.balanced_accuracy, R[m].skin?.balanced_accuracy_ci)}</td>
                  <td className="py-3 pr-4">{fixed(R[m].skin?.macro_auroc)}</td>
                  <td className="py-3 pr-4">{fixed(R[m].brain?.accuracy)}</td>
                  <td className="py-3 pr-4">{fixed(R[m].brain?.macro_auroc)}</td>
                  <td className="py-3">{fixed(R[m].ece)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-4 text-xs text-ink-muted">95% bootstrap intervals in brackets. Thresholds and temperatures were fitted on validation only.</p>
      </Section>

      <div className="grid items-start gap-6 xl:grid-cols-2">
        <Section n="02" highlight="Chest" title="AUROC per finding">
          <div className="mb-4">
            <ModeLegend modes={['multimodal', 'image_only'].filter((m) => R[m])} />
          </div>
          <PerClassDots results={R} />
        </Section>
        <Section n="03" highlight="Training" title="curve">
          <p className="mb-4 text-xs text-ink-muted">Validation AUROC, all 25 classes pooled, per epoch. The best epoch was kept.</p>
          <div className="mb-4">
            <ModeLegend modes={modes} />
          </div>
          <TrainingCurve history={info.training_history} />
        </Section>
      </div>

      <div className="grid items-start gap-6 lg:grid-cols-2">
        <Section n="04" highlight="Data" title="splits">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-line text-left text-[11px] uppercase tracking-[0.12em] text-ink-muted">
                <th className="py-2.5 font-medium">Split</th>
                {Object.keys(MODALITIES).map((k) => (
                  <th key={k} className="py-2.5 text-right font-medium">
                    {MODALITIES[k].short}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="num">
              {['train', 'val', 'test'].map((s) => (
                <tr key={s} className="border-b border-line text-ink-2">
                  <td className="py-3 capitalize text-ink">{s}</td>
                  {Object.keys(MODALITIES).map((k) => (
                    <td key={k} className="py-3 text-right">
                      {counts[s]?.[k]?.toLocaleString() ?? '–'}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <dl className="mt-6 grid grid-cols-2 gap-5 text-sm">
            <div>
              <dt className="text-[11px] uppercase tracking-[0.12em] text-ink-muted">Image encoder</dt>
              <dd className="mt-1 text-ink">{info.image_backbone}</dd>
            </div>
            <div>
              <dt className="text-[11px] uppercase tracking-[0.12em] text-ink-muted">Text encoder</dt>
              <dd className="mt-1 truncate text-ink" title={info.text_model_name}>
                {info.text_model_name}
              </dd>
            </div>
            <div>
              <dt className="text-[11px] uppercase tracking-[0.12em] text-ink-muted">Chest training patients</dt>
              <dd className="num mt-1 text-ink">{pct(info.chest_train_fraction, 0)}</dd>
            </div>
            <div>
              <dt className="text-[11px] uppercase tracking-[0.12em] text-ink-muted">Modality router accuracy</dt>
              <dd className="num mt-1 text-ink">{pct(info.modality_classifier_test_accuracy, 2)}</dd>
            </div>
          </dl>
        </Section>
        <Section n="05" highlight="Protocol" title="for evaluation">
          <dl className="space-y-5 text-sm">
            {Object.entries(info.evaluation_protocol || {}).map(([k, v]) => (
              <div key={k} className="flex gap-3">
                <span className="tri mt-1.5 rotate-90" style={{ borderBottomColor: '#a99ff0' }} />
                <div>
                  <dt className="capitalize text-ink">{k}</dt>
                  <dd className="mt-0.5 leading-relaxed text-ink-muted">{v}</dd>
                </div>
              </div>
            ))}
          </dl>
        </Section>
      </div>
    </div>,
  )
}
