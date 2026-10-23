// Numbered glass strip: "01  Findings <rest of title>"
export default function Section({ n, highlight, title, aside, children, id, className = '' }) {
  return (
    <section id={id} className={`glass scroll-mt-28 px-5 py-6 sm:px-8 sm:py-7 print-ink ${className}`}>
      <div className="mb-6 flex flex-wrap items-center gap-x-6 gap-y-3">
        <span className="display num text-4xl text-violet">{n}</span>
        <h2 className="display text-2xl sm:text-[28px]">
          <span className="text-violet">{highlight}</span> {title}
        </h2>
        {aside && <div className="flex flex-wrap gap-2 sm:ml-auto">{aside}</div>}
      </div>
      {children}
    </section>
  )
}
