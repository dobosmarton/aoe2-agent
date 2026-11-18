// GLSL shared by the scene's custom materials.

/** Ends a fragment shader the way three's built-in materials do: tone mapping, then the output colour space. */
export const GLSL_OUTPUT_COLOR = /* glsl */ `
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
`;
