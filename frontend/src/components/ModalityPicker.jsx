import { MODALITIES } from '../lib/domain'

const OPTIONS = [['auto', 'Auto'], ...Object.entries(MODALITIES).map(([k, v]) => [k, v.short])]

export default function ModalityPicker({ value, onChange }) {
  return (
    <section className="card card-pad">
      <div className="mb-3 flex items-baseline justify-between gap-3">
        <h2 className="card-title">Modality</h2>
        <span className="text-right text-xs text-ink-muted">
          {value === 'auto' ? 'a classifier routes the image' : MODALITIES[value].dataset}
        </span>
      </div>
      <div className="seg" role="radiogroup" aria-label="Modality">
        {OPTIONS.map(([key, label]) => (
          <button
            key={key}
            type="button"
            role="radio"
            aria-checked={value === key}
            onClick={() => onChange(key)}
            className={`seg-btn ${value === key ? 'seg-btn-on' : ''}`}
          >
            {label}
          </button>
        ))}
      </div>
    </section>
  )
}
