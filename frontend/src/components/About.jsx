import { Github } from './Icons'
import Section from './Section'

const REPO = 'https://github.com/ARRY7686/Multi-Modal-Disease-Detection-System'

export default function About() {
  return (
    <div className="space-y-6 pt-10 lg:pt-16">
      <div className="pb-6">
        <div className="eyebrow mb-4">About the project</div>
        <h1 className="display max-w-4xl text-[40px] leading-[1.08] sm:text-[52px]">
          One network, three kinds of scan, <span className="text-violet">and the numbers it actually earns</span>
        </h1>
      </div>

      <Section n="01" highlight="What" title="it is">
        <p className="max-w-3xl text-[15px] leading-relaxed text-ink-2">
          A single model that reads a medical image (chest X-ray, dermoscopy photo or brain MRI) together with the patient
          details recorded alongside it, and scores 25 findings. EfficientNet-B4 encodes the image, BioBERT encodes the
          text, and two layers of bidirectional cross-attention fuse them. Image-only and text-only heads are trained in
          the same run, so every mode is a real trained model.
        </p>
      </Section>

      <Section n="02" highlight="Why" title="the first numbers were thrown away">
        <div className="grid max-w-5xl gap-8 text-[15px] leading-relaxed text-ink-2 md:grid-cols-2">
          <p>
            Version 1 reported a multimodal AUROC of 0.989. That measured leakage, not diagnosis: the paired text was
            generated from each image&apos;s label, so the model could read the answer from the text (text alone scored
            0.999), and the same patient, lesion or duplicated MRI could land in both training and test.
          </p>
          <p>
            The rebuilt pipeline pairs each image with its real recorded metadata, splits by patient, lesion and duplicate
            cluster, fits thresholds and temperatures on validation only, and scores the test sets once. A test fails if any
            label word appears in the paired text.
          </p>
        </div>
      </Section>

      <Section n="03" highlight="Reading" title="the outputs">
        <ul className="max-w-3xl space-y-4 text-[15px] text-ink-2">
          {[
            'Scores are per-class probabilities (multi-label), temperature-scaled on validation data.',
            'A class is flagged when its score reaches its own threshold, fitted on validation to maximise F1.',
            'Grad-CAM and attention show what the network used, which is not the same as where the disease is.',
          ].map((t) => (
            <li key={t} className="flex gap-3">
              <span className="tri mt-2 rotate-90" style={{ borderBottomColor: '#a99ff0' }} />
              {t}
            </li>
          ))}
        </ul>
      </Section>

      <div className="glass flex flex-wrap items-center gap-6 border-status-critical/30 p-6 sm:p-8">
        <p className="max-w-3xl flex-1 text-[15px] leading-relaxed text-ink-2">
          <span className="text-ink">Research prototype, not a medical device.</span> Trained on public research datasets,
          not clinically validated, and not for diagnosis or treatment decisions.
        </p>
        <a href={REPO} target="_blank" rel="noreferrer" className="pill">
          <Github className="h-3.5 w-3.5" /> Source and training pipeline
        </a>
      </div>
    </div>
  )
}
