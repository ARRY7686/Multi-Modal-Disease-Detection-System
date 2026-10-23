import { MODALITIES, prettyLabel } from './domain'

// Example cases: held-out test images with precomputed outputs, written by
// scripts/build_gallery.py into public/gallery/. The static build
// (`npm run build:static`) has no backend and answers from these files.
export const STATIC = import.meta.env.VITE_STATIC === '1'

const BASE = `${import.meta.env.BASE_URL}gallery/`
export const galleryUrl = (name) => BASE + name

export async function galleryJson(name) {
  const res = await fetch(galleryUrl(name))
  if (!res.ok) throw new Error(`gallery/${name}: HTTP ${res.status}`)
  return res.json()
}

let index = null
/** The example cases, or [] when the gallery has not been built. */
export function loadGallery() {
  index =
    index ||
    galleryJson('index.json')
      .then((d) => d.cases || [])
      .catch(() => [])
  return index
}

export async function exampleFile(example) {
  const blob = await (await fetch(galleryUrl(example.image))).blob()
  return new File([blob], example.image, { type: blob.type || 'image/jpeg' })
}

/** "Melanoma", "Cardiomegaly", "No finding" / "No tumor" for a case's label list. */
export function exampleLabel(example) {
  if (example.label.length) return example.label.map(prettyLabel).join(', ')
  return example.modality === 'brain' ? 'No tumor' : 'No finding'
}

export const exampleTitle = (example) => `${MODALITIES[example.modality].short}: ${exampleLabel(example)}`
