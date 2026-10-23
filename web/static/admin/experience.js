/* ScholarGraph ambient field — intentionally one lightweight Three.js scene. */
import * as THREE from '/static/vendor/three.module.min.js';

const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const canvas = document.getElementById('constellation');
const orb = document.getElementById('thinkingOrb');
let renderer, scene, camera, points, raf;

function setThinking(active) {
  if (!orb) return;
  orb.classList.toggle('is-thinking', Boolean(active));
  orb.setAttribute('title', active ? 'ScholarGraph is reasoning' : 'Research intelligence standby');
}

function field() {
  if (!canvas || reduce || !window.WebGLRenderingContext) return;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: false, powerPreference: 'low-power' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(54, 1, 0.1, 100);
    camera.position.z = 24;
    const mobile = innerWidth < 700;
    const count = mobile ? 420 : 1150; // stays safely below the 3,000-particle budget.
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const violet = new THREE.Color('#9b7bff');
    const cyan = new THREE.Color('#62f0e3');
    for (let i = 0; i < count; i += 1) {
      const n = i * 3, edge = Math.pow(Math.random(), .68);
      positions[n] = (Math.random() - .5) * 55;
      positions[n + 1] = (Math.random() - .5) * 38;
      positions[n + 2] = (Math.random() - .5) * 20;
      const c = violet.clone().lerp(cyan, edge);
      colors[n] = c.r; colors[n + 1] = c.g; colors[n + 2] = c.b;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    points = new THREE.Points(geometry, new THREE.PointsMaterial({ size: mobile ? .11 : .08, vertexColors: true, transparent: true, opacity: .52, sizeAttenuation: true }));
    scene.add(points);
    const resize = () => {
      const { innerWidth: w, innerHeight: h } = window;
      renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
    };
    window.addEventListener('resize', resize, { passive: true }); resize();
    const tick = (t) => { points.rotation.y = t * .000018; points.rotation.x = Math.sin(t * .00007) * .045; renderer.render(scene, camera); raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick);
  } catch (_) { canvas.closest('.ambient')?.classList.add('no-webgl'); }
}

function lottiePulse() {
  if (!window.lottie || !document.getElementById('lottiePulse')) return;
  window.lottie.loadAnimation({
    container: document.getElementById('lottiePulse'), renderer: 'svg', loop: true, autoplay: true,
    animationData: { v: '5.7.4', fr: 30, ip: 0, op: 90, w: 40, h: 40, nm: 'signal', layers: [{ ty: 4, ks: { o: { a: 0, k: 70 }, r: { a: 0, k: 0 }, p: { a: 0, k: [20,20,0] }, a: { a: 0, k: [0,0,0] }, s: { a: 1, k: [{ t: 0, s: [25,25,100] }, { t: 45, s: [100,100,100] }, { t: 90, s: [25,25,100] }] } }, shapes: [{ ty: 'el', p: { a: 0, k: [0,0] }, s: { a: 0, k: [28,28] } }, { ty: 'st', c: { a: 0, k: [0.39,0.94,0.89,1] }, o: { a: 0, k: 100 }, w: { a: 0, k: 1.6 } }], ip: 0, op: 90, st: 0 }] }
  });
}

function motion() {
  if (reduce || !window.gsap) return;
  window.gsap.from('.brand, .topbar', { y: -14, opacity: 0, duration: .72, ease: 'power3.out', stagger: .08 });
  window.gsap.from('.rail .nav button', { x: -14, opacity: 0, duration: .4, ease: 'power2.out', stagger: .045, delay: .18 });
  const observer = new MutationObserver(() => {
    const view = document.querySelector('.view.active');
    if (view && !view.dataset.revealed) { view.dataset.revealed = 'true'; window.gsap.from(view.querySelectorAll('.card, .topic-card, .exp-row'), { opacity: 0, y: 14, duration: .42, stagger: .035, ease: 'power2.out' }); }
  });
  observer.observe(document.querySelector('.content'), { subtree: true, childList: true, attributes: true, attributeFilter: ['class'] });
}

const poll = window.setInterval(() => setThinking(document.getElementById('runStatus')?.classList.contains('running')), 650);
window.addEventListener('beforeunload', () => { cancelAnimationFrame(raf); clearInterval(poll); renderer?.dispose(); });
if (window.lucide) window.lucide.createIcons();
field(); lottiePulse(); motion();
