import axios from 'axios'

// Same origin when the API serves the built app (Docker / HF Space): build with
// VITE_API_URL="" . Local dev falls back to the uvicorn default.
const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

const api = axios.create({ baseURL: API_BASE, timeout: 120000 })

export async function checkHealth() {
  const { data } = await api.get('/health')
  return data
}

export async function getModelInfo() {
  const { data } = await api.get('/model-info')
  return data
}

/** image (File) and/or symptoms; modality 'auto' lets the server decide. */
export async function predict({ image, symptoms, modality }) {
  const form = new FormData()
  if (image) form.append('image', image)
  if (symptoms) form.append('symptoms', symptoms)
  if (modality && modality !== 'auto') form.append('modality', modality)
  const { data } = await api.post('/predict', form)
  return data
}

export function errorMessage(err) {
  const detail = err?.response?.data?.detail
  // FastAPI validation errors are a list of {loc, msg} objects
  const message = Array.isArray(detail) ? detail.map((d) => d.msg).join('; ') : detail
  if (message) return message
  if (err?.code === 'ECONNABORTED') return 'The request timed out. The server may still be loading the models.'
  if (!err?.response) return 'Cannot reach the API. Is the server running?'
  return err.message || 'Prediction failed.'
}
