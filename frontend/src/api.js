import axios from 'axios'
import { STATIC, galleryJson } from './lib/gallery'

// Same origin when the API serves the built app (Docker / HF Space): build with
// VITE_API_URL="" . Local dev falls back to the uvicorn default.
const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

const api = axios.create({ baseURL: API_BASE, timeout: 120000 })

/** Raised by the static demo when it has no precomputed answer. */
export class DemoError extends Error {}

export async function checkHealth() {
  if (STATIC) return galleryJson('health.json')
  const { data } = await api.get('/health')
  return data
}

export async function getModelInfo() {
  if (STATIC) return galleryJson('model-info.json')
  const { data } = await api.get('/model-info')
  return data
}

// Static demo: answer from the example's cached responses when the inputs are
// still the example's own (its image, its details, or both).
async function cachedPredict({ image, symptoms, example }) {
  if (!example || (image && image.name !== example.image)) {
    throw new DemoError('This demo has no backend, so it can only show the example cases. Run the app locally or with Docker to analyze your own scan.')
  }
  if (symptoms && symptoms !== example.text) {
    throw new DemoError(
      `Results were computed with this example's own details ("${example.text}"). Pick the example again to restore them, or run the full app to try other details.`,
    )
  }
  const { results } = await galleryJson(`${example.id}.json`)
  return results[image && symptoms ? 'multimodal' : image ? 'image_only' : 'text_only']
}

/**
 * image (File) and/or symptoms; modality 'auto' lets the server decide.
 * `example` is the gallery case the inputs came from (used by the static demo).
 */
export async function predict({ image, symptoms, modality, example }) {
  if (STATIC) return cachedPredict({ image, symptoms, example })
  const form = new FormData()
  if (image) form.append('image', image)
  if (symptoms) form.append('symptoms', symptoms)
  if (modality && modality !== 'auto') form.append('modality', modality)
  const { data } = await api.post('/predict', form)
  return data
}

export function errorMessage(err) {
  if (err instanceof DemoError) return err.message
  const detail = err?.response?.data?.detail
  // FastAPI validation errors are a list of {loc, msg} objects
  const message = Array.isArray(detail) ? detail.map((d) => d.msg).join('; ') : detail
  if (message) return message
  if (err?.code === 'ECONNABORTED') return 'The request timed out. The server may still be loading the models.'
  if (!err?.response) return 'Cannot reach the API. Is the server running?'
  return err.message || 'Prediction failed.'
}
