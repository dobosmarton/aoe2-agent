import * as THREE from "three";

// The camera is an orbit rig: a target, an azimuth and elevation around it, the
// half-height of the view at the target (`size`) and a field of view. Distance follows
// from size and fov, so narrowing the fov at a fixed size is a dolly zoom that flattens
// the view towards the game's orthographic camera without changing the framing.
export type CameraPose = {
  tx: number;
  ty: number;
  tz: number;
  /** Degrees; 45 puts the camera on the +x +z side, like the game camera. */
  az: number;
  /** Degrees above the ground plane. */
  el: number;
  size: number;
  fov: number;
};

const target = new THREE.Vector3();

export function applyPose(camera: THREE.PerspectiveCamera, pose: CameraPose, aspect: number, fit: number) {
  const fov = THREE.MathUtils.clamp(pose.fov, 2, 90);
  const size = pose.size * fit;
  const distance = size / Math.tan(THREE.MathUtils.degToRad(fov / 2));
  const az = THREE.MathUtils.degToRad(pose.az);
  const el = THREE.MathUtils.degToRad(pose.el);
  target.set(pose.tx, pose.ty, pose.tz);
  // Azimuth is measured from +x towards +z.
  camera.position.set(
    target.x + distance * Math.cos(el) * Math.cos(az),
    target.y + distance * Math.sin(el),
    target.z + distance * Math.cos(el) * Math.sin(az),
  );
  camera.fov = fov;
  camera.aspect = aspect;
  camera.near = Math.max(0.05, distance * 0.02);
  camera.far = distance * 6 + 200;
  camera.up.set(0, 1, 0);
  camera.lookAt(target);
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld();
  return distance;
}
