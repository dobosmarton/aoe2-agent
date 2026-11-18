import * as THREE from "three";
import {
  BloomEffect,
  EffectComposer,
  BlendFunction,
  EffectPass,
  NoiseEffect,
  RenderPass,
  SMAAEffect,
  SMAAPreset,
  TiltShiftEffect,
  ToneMappingEffect,
  ToneMappingMode,
  VignetteEffect,
} from "postprocessing";

// One pmndrs/postprocessing chain: a single merged effect pass with tilt-shift (the
// miniature look of the world), bloom (the glow of the mind), tone mapping and a
// vignette, then SMAA. Phones skip the tilt-shift. Neutral tone mapping keeps the
// game's sprite colours close to how the game draws them.
//
// Cost control: the blur chains behind tilt-shift and bloom only run while they show
// (a 2x frame spends about 0.7 ms and 1 ms on them), and SMAA (about 1.6 ms) only runs
// below 1.75x, where edges still step visibly.
const SMAA_BELOW_DPR = 1.75;

export type Post = {
  setSize: (width: number, height: number) => void;
  /** 0..1: how strongly the world reads as a miniature. */
  setMiniature: (amount: number) => void;
  /** Bloom strength; 0 when nothing in view is bright enough to bloom. */
  setBloom: (amount: number) => void;
  render: (delta: number) => void;
  dispose: () => void;
};

type SkippableEffect = { update: TiltShiftEffect["update"] };

// Skip an effect's own passes (its blur) while it contributes nothing. Its shader still
// runs but weighs that stale result by zero; the first frame it shows again recomputes it.
const skipWhile = (effect: SkippableEffect, idle: () => boolean): void => {
  const update = effect.update.bind(effect);

  effect.update = (r, input, delta) => {
    if (!idle()) {
      update(r, input, delta);
    }
  };
};

/** `small`: phones, which skip the tilt-shift. */
export const createPost = (renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.Camera, small: boolean): Post => {
  const composer = new EffectComposer(renderer, { frameBufferType: THREE.HalfFloatType });

  composer.addPass(new RenderPass(scene, camera));

  // A gentle miniature: only the top and bottom bands of a wide shot soften.
  const tiltShift = new TiltShiftEffect({ focusArea: 0.7, feather: 0.28, offset: 0.05, resolutionScale: 0.5 });
  // Only colours pushed past 1 (the mind's lines and glows) bloom; sprites never do.
  const bloom = new BloomEffect({ intensity: 0.9, luminanceThreshold: 1.0, luminanceSmoothing: 0.25, mipmapBlur: true, radius: 0.65 });
  const vignette = new VignetteEffect({ offset: 0.42, darkness: 0.32 });
  const tone = new ToneMappingEffect({ mode: ToneMappingMode.NEUTRAL });
  // Fine grain against banding in the dark gradients.
  const grain = new NoiseEffect({ blendFunction: BlendFunction.OVERLAY, premultiply: false });

  grain.blendMode.opacity.value = 0.06;

  const effects = small ? [bloom, tone, vignette, grain] : [tiltShift, bloom, tone, vignette, grain];

  composer.addPass(new EffectPass(camera, ...effects));

  const smaa = new EffectPass(camera, new SMAAEffect({ preset: SMAAPreset.HIGH }));

  composer.addPass(smaa);

  skipWhile(tiltShift, () => tiltShift.blendMode.opacity.value < 0.002);
  skipWhile(bloom, () => bloom.intensity < 0.002);

  return {
    setSize: (width, height) => {
      composer.setSize(width, height);
      smaa.enabled = renderer.getPixelRatio() < SMAA_BELOW_DPR;

      // The last enabled pass draws to the screen.
      composer.passes.forEach((pass, i, all) => {
        pass.renderToScreen = all.slice(i + 1).every((p) => !p.enabled);
      });
    },
    setMiniature: (amount) => {
      tiltShift.blendMode.opacity.value = amount;
    },
    setBloom: (amount) => {
      bloom.intensity = amount;
    },
    render: (delta) => composer.render(delta),
    dispose: () => composer.dispose(),
  };
};
