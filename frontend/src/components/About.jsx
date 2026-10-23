import { Github } from './Icons'

const REPO = 'https://github.com/ARRY7686/Multi-Modal-Disease-Detection-System'

function Block({ title, children }) {
  return (
    <section className="card card-pad">
      <h2 className="card-title">{title}</h2>
      <div className="mt-3 space-y-3 text-sm leading-relaxed text-ink-2">{children}</div>
    </section>
  )
}

export default function About() {
  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <Block title="What this is">
        <p>
          One network that reads a medical image (chest X-ray, dermoscopy photo or brain MRI) together with the
          patient details recorded alongside it, and scores 25 findings. An EfficientNet-B4 encodes the image, BioBERT
          encodes the text, and two layers of bidirectional cross-attention fuse them. Image-only and text-only heads
          are trained in the same run, so every mode is a real trained model.
        </p>
      </Block>
      <Block title="Why the first version's numbers were thrown away">
        <p>
          Version 1 reported a multimodal AUROC of 0.989. That measured leakage, not diagnosis: the paired text was
          generated from each image&apos;s label, so the model could read the answer from the text (text alone scored
          0.999), and the same patient, lesion or duplicated MRI could land in both training and test.
        </p>
        <p>
          The rebuilt pipeline pairs each image with its real recorded metadata, splits by patient, lesion and duplicate
          cluster, fits thresholds and temperatures on validation only, and scores the test sets once. A test in the
          repository fails if any label word appears in the paired text. The Model card tab shows the honest results.
        </p>
      </Block>
      <Block title="How to read the outputs">
        <ul className="list-disc space-y-2 pl-5">
          <li>Scores are per-class probabilities (multi-label), temperature-scaled on validation data.</li>
          <li>A class is flagged when its score reaches its own threshold, fitted on validation to maximise F1.</li>
          <li>Grad-CAM and attention show what the network used, which is not the same as where the disease is.</li>
        </ul>
      </Block>
      <div className="rounded-2xl border border-status-critical/40 bg-status-critical/5 p-5 text-sm leading-relaxed text-ink-2">
        <strong className="text-ink">Research prototype, not a medical device.</strong> It was trained on public
        research datasets, has not been clinically validated, and must not be used for diagnosis or treatment decisions.
      </div>
      <a href={REPO} target="_blank" rel="noreferrer" className="btn-ghost w-fit">
        <Github /> Source code and training pipeline
      </a>
    </div>
  )
}
