import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 60000,
})

export async function checkHealth() {
  const { data } = await api.get('/health')
  return data
}

export async function getLabels() {
  const { data } = await api.get('/labels')
  return data
}

export async function predict({ image, symptoms, modality }) {
  const formData = new FormData()
  if (image) formData.append('image', image)
  if (symptoms) formData.append('symptoms', symptoms)
  if (modality && modality !== 'auto') formData.append('modality', modality)

  const { data } = await api.post('/predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function predictMultimodal({ image, symptoms, modality }) {
  const formData = new FormData()
  formData.append('image', image)
  formData.append('symptoms', symptoms)
  if (modality && modality !== 'auto') formData.append('modality', modality)

  const { data } = await api.post('/predict-multimodal', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}
