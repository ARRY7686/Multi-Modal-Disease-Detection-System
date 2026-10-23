// Labels and wording shared by the UI. The patient-detail sentences mirror
// training/data.py exactly: the model only ever saw these sentence shapes.

export const MODALITIES = {
  chest: { label: 'Chest X-ray', short: 'Chest', dataset: 'NIH ChestX-ray14' },
  skin: { label: 'Skin lesion', short: 'Skin', dataset: 'HAM10000 dermoscopy' },
  brain: { label: 'Brain MRI', short: 'Brain', dataset: 'Brain tumour MRI' },
}

export const MODES = {
  multimodal: { label: 'Multimodal', color: '#9085e9', blurb: 'image + details, cross-attention fusion' },
  image_only: { label: 'Image only', color: '#d95926', blurb: 'EfficientNet-B4 head' },
  text_only: { label: 'Details only', color: '#199e70', blurb: 'BioBERT head on the patient details' },
}
export const MODE_ORDER = ['multimodal', 'image_only', 'text_only']

export const SOURCE_LABELS = {
  user: 'chosen manually',
  classifier: 'auto-detected',
  heuristic: 'auto-detected (colour heuristic)',
  text: 'inferred from the text',
}

// HAM10000 `localization` values (minus "unknown")
export const LESION_SITES = [
  'abdomen', 'acral', 'back', 'chest', 'ear', 'face', 'foot', 'genital', 'hand',
  'lower extremity', 'neck', 'scalp', 'trunk', 'upper extremity',
]

export const NO_DETAILS = 'No clinical details provided.'

export const prettyLabel = (s) => (s || '').replace(/_/g, ' ')

function patientSentence(age, sex) {
  const a = Number(age)
  const words = [a > 0 && a <= 100 ? `${Math.round(a)}-year-old` : null, sex || null].filter(Boolean)
  if (!words.length) return null
  const who = words.join(' ')
  return `${who[0].toUpperCase()}${who.slice(1)}.`
}

/** Same sentence shapes as training/data.py (chest_text / skin_text / brain). */
export function buildDetailsText({ modality, age, sex, view, site }) {
  const parts = [patientSentence(age, sex)]
  if (modality === 'chest' && (view === 'PA' || view === 'AP')) parts.push(`${view} chest radiograph.`)
  if (modality === 'skin' && site) parts.push(`Lesion on the ${site}.`)
  if (modality === 'brain') return NO_DETAILS
  return parts.filter(Boolean).join(' ') || ''
}

/**
 * When an analysis ran both ways and the patient details changed the top
 * finding: {image, fused} predictions, else null.
 */
export function detailsOverride(results) {
  const image = results?.image_only?.predictions?.[0]
  const fused = results?.multimodal?.predictions?.[0]
  if (!image || !fused || image.label_index === fused.label_index) return null
  return { image, fused }
}

export const pct = (x, digits = 1) => (x == null || Number.isNaN(x) ? '–' : `${(x * 100).toFixed(digits)}%`)
export const fixed = (x, digits = 3) => (x == null || Number.isNaN(x) ? '–' : Number(x).toFixed(digits))
