// GLSL shared by the scene's custom materials.

/** Passes the plane's uv to the fragment shader as `vUv`. */
export const GLSL_UV_VERTEX = /* glsl */ `
  varying vec2 vUv;
  void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
`;

/** Ends a fragment shader the way three's built-in materials do: tone mapping, then the output colour space. */
export const GLSL_OUTPUT_COLOR = /* glsl */ `
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
`;
