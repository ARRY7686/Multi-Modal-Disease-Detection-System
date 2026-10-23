import { MODALITIES } from '../lib/domain'

const OPTIONS = [['auto', 'Auto'], ...Object.entries(MODALITIES).map(([k, v]) => [k, v.short])]

export default function ModalityPicker({ value, onChange }) {
  return (
    <div>
      <div className="mb-2.5 flex items-baseline justify-between gap-3">
        <span className="eyebrow">Modality</span>
        <span className="text-right text-xs text-ink-muted">
          {value === 'auto' ? 'a classifier routes the image' : MODALITIES[value].dataset}
        </span>
      </div>
      <div className="toggle-group flex w-full" role="radiogroup" aria-label="Modality">
        {OPTIONS.map(([key, label]) => (
          <button
            key={key}
            type="button"
            role="radio"
            aria-checked={value === key}
            onClick={() => onChange(key)}
            className={`toggle flex-1 ${value === key ? 'toggle-on' : ''}`}
          >
            {label}
          </button>
        ))}
      </div>
    </div>
  )
}
