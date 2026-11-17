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
export function createPost(renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.Camera, small: boolean) {
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
  composer.addPass(new EffectPass(camera, new SMAAEffect({ preset: SMAAPreset.HIGH })));

  return {
    composer,
    setSize(width: number, height: number) {
      composer.setSize(width, height);
    },
    /** 0..1: how strongly the world reads as a miniature. */
    setMiniature(amount: number) {
      tiltShift.blendMode.opacity.value = amount;
    },
    setBloom(amount: number) {
      bloom.intensity = amount;
    },
    render(delta: number) {
      composer.render(delta);
    },
    dispose() {
      composer.dispose();
    },
  };
}
