import * as THREE from "three";
import {
  BloomEffect,
  EffectComposer,
  EffectPass,
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

  const tiltShift = new TiltShiftEffect({ focusArea: 0.56, feather: 0.34, offset: 0.04, resolutionScale: 0.5 });
  const bloom = new BloomEffect({ intensity: 0.9, luminanceThreshold: 0.85, luminanceSmoothing: 0.2, mipmapBlur: true, radius: 0.7 });
  const vignette = new VignetteEffect({ offset: 0.32, darkness: 0.5 });
  const tone = new ToneMappingEffect({ mode: ToneMappingMode.NEUTRAL });
  const effects = small ? [bloom, tone, vignette] : [tiltShift, bloom, tone, vignette];
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
