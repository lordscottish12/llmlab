// A Three.js stage with orbit controls and HTML labels, sized to its container and theme-aware.

import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/CSS2DRenderer.js";
import { cssVar } from "./ui.js";

export { THREE };

export function createStage(container, { camera = [3, 2.2, 3.4], target = [0, 0, 0], fov = 45 } = {}) {
  const scene = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(fov, 1, 0.01, 500);
  cam.position.set(...camera);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  container.append(renderer.domElement);
  const labels = new CSS2DRenderer();
  labels.domElement.className = "labels";
  container.append(labels.domElement);
  const controls = new OrbitControls(cam, renderer.domElement);
  controls.target.set(...target);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  scene.add(new THREE.AmbientLight(0xffffff, 1.6));
  const sun = new THREE.DirectionalLight(0xffffff, 1.4);
  sun.position.set(3, 6, 4);
  scene.add(sun);

  const stage = {
    scene, camera: cam, renderer, controls, labels, THREE,
    onFrame: [],
    color: (name) => new THREE.Color(cssVar(name) || "#888"),
    label(text, cls, pos) {
      const el = document.createElement("div");
      el.className = cls;
      if (text instanceof Node) el.append(text); else el.textContent = text;
      const obj = new CSS2DObject(el);
      obj.position.set(...pos);
      return obj;
    },
    clear(group) {
      for (const c of [...group.children]) {
        group.remove(c);
        c.traverse?.((o) => {
          o.geometry?.dispose?.();
          if (o.material) [].concat(o.material).forEach((m) => { m.map?.dispose?.(); m.dispose?.(); });
          if (o.element) o.element.remove();
        });
      }
    },
    dispose() {
      cancelAnimationFrame(raf);
      ro.disconnect();
      mo.disconnect();
      controls.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      labels.domElement.remove();
    },
    onTheme: [],
  };

  const resize = () => {
    const w = container.clientWidth, hgt = container.clientHeight;
    if (!w || !hgt) return;
    renderer.setSize(w, hgt, false);
    renderer.domElement.style.width = `${w}px`;
    renderer.domElement.style.height = `${hgt}px`;
    labels.setSize(w, hgt);
    cam.aspect = w / hgt;
    cam.updateProjectionMatrix();
  };
  const ro = new ResizeObserver(resize);
  ro.observe(container);
  resize();

  // Re-colour when Day/Night changes.
  const mo = new MutationObserver(() => stage.onTheme.forEach((f) => f()));
  mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

  let raf;
  const loop = (t) => {
    raf = requestAnimationFrame(loop);
    controls.update();
    stage.onFrame.forEach((f) => f(t));
    renderer.render(scene, cam);
    labels.render(scene, cam);
  };
  raf = requestAnimationFrame(loop);
  return stage;
}

// A thin arrow from a to b (both [x,y,z]).
export function arrow(THREE, a, b, color, { radius = 0.008, head = 0.06 } = {}) {
  const from = new THREE.Vector3(...a), to = new THREE.Vector3(...b);
  const dir = to.clone().sub(from);
  const len = dir.length();
  const g = new THREE.Group();
  if (len < 1e-6) return g;
  const mat = new THREE.MeshBasicMaterial({ color });
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, Math.max(0, len - head), 8), mat);
  shaft.position.y = (len - head) / 2;
  const tip = new THREE.Mesh(new THREE.ConeGeometry(radius * 3.2, head, 12), mat);
  tip.position.y = len - head / 2;
  g.add(shaft, tip);
  g.position.copy(from);
  g.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
  return g;
}
