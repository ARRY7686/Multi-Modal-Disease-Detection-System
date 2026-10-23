// Inline stroke icons (24px grid), sized and coloured by the parent.
const base = { fill: 'none', stroke: 'currentColor', strokeWidth: 1.8, strokeLinecap: 'round', strokeLinejoin: 'round' }
const Icon = ({ d, className = 'h-4 w-4', children }) => (
  <svg viewBox="0 0 24 24" className={className} aria-hidden="true" {...base}>
    {d ? <path d={d} /> : children}
  </svg>
)

export const Pulse = (p) => <Icon {...p} d="M3 12h4l2.5-6 4 12 2.5-6H21" />
export const Upload = (p) => <Icon {...p} d="M12 16V4m0 0-4 4m4-4 4 4M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" />
export const Close = (p) => <Icon {...p} d="M6 6l12 12M18 6 6 18" />
export const Spark = (p) => <Icon {...p} d="M12 3v4m0 10v4M3 12h4m10 0h4M6 6l2.5 2.5m7 7L18 18M6 18l2.5-2.5m7-7L18 6" />
export const Alert = (p) => <Icon {...p} d="M12 9v4m0 4h.01M10.3 3.9 2.4 18a2 2 0 0 0 1.7 3h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z" />
export const Clock = (p) => <Icon {...p}><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></Icon>
export const Download = (p) => <Icon {...p} d="M12 4v12m0 0-4-4m4 4 4-4M4 20h16" />
export const Printer = (p) => <Icon {...p} d="M7 9V3h10v6M7 17H5a2 2 0 0 1-2-2v-4a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v4a2 2 0 0 1-2 2h-2M7 14h10v7H7z" />
export const Expand = (p) => <Icon {...p} d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" />
export const Layers = (p) => <Icon {...p} d="m12 3 9 5-9 5-9-5 9-5Zm-9 9 9 5 9-5M3 16l9 5 9-5" />
export const Info = (p) => <Icon {...p}><circle cx="12" cy="12" r="9" /><path d="M12 11v5m0-8h.01" /></Icon>
export const Github = (p) => (
  <svg viewBox="0 0 24 24" className={p.className || 'h-4 w-4'} aria-hidden="true" fill="currentColor">
    <path d="M12 2a10 10 0 0 0-3.16 19.49c.5.09.68-.22.68-.48v-1.7c-2.78.6-3.37-1.34-3.37-1.34-.45-1.15-1.11-1.46-1.11-1.46-.91-.62.07-.6.07-.6 1 .07 1.53 1.03 1.53 1.03.9 1.52 2.34 1.08 2.91.83.09-.65.35-1.08.63-1.33-2.22-.25-4.55-1.11-4.55-4.94 0-1.09.39-1.98 1.03-2.68-.1-.25-.45-1.27.1-2.64 0 0 .84-.27 2.75 1.02a9.6 9.6 0 0 1 5 0c1.91-1.29 2.75-1.02 2.75-1.02.55 1.37.2 2.39.1 2.64.64.7 1.03 1.59 1.03 2.68 0 3.84-2.34 4.68-4.57 4.93.36.31.68.92.68 1.85v2.74c0 .27.18.58.69.48A10 10 0 0 0 12 2Z" />
  </svg>
)
