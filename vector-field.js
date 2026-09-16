(() => {
  // Replace the earlier grid/cursor effect with a quieter research-style vector field.
  const style = document.createElement('style');
  style.textContent = `
    .sciml-field, .cursor-orbit { display: none !important; }
    .vector-field-canvas {
      position: fixed !important;
      inset: 0 !important;
      width: 100% !important;
      height: 100% !important;
      pointer-events: none !important;
      z-index: 0 !important;
      opacity: .9;
    }
    body > *:not(.vector-field-canvas):not(.theme-switcher) {
      position: relative;
      z-index: 1;
    }
    @media (prefers-reduced-motion: reduce) {
      .vector-field-canvas { opacity: .45; }
    }
  `;
  document.head.appendChild(style);

  const canvas = document.createElement('canvas');
  canvas.className = 'vector-field-canvas';
  canvas.setAttribute('aria-hidden', 'true');
  document.body.prepend(canvas);

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const finePointer = window.matchMedia('(pointer: fine)').matches;

  let width = 0;
  let height = 0;
  let dpr = 1;
  let raf = 0;
  let visible = true;

  const mouse = {
    x: window.innerWidth * 0.58,
    y: window.innerHeight * 0.40,
    tx: window.innerWidth * 0.58,
    ty: window.innerHeight * 0.40,
    active: false
  };

  let impulse = 0;
  let impulseX = mouse.x;
  let impulseY = mouse.y;
  let lastTime = performance.now();

  function resize() {
    width = window.innerWidth;
    height = window.innerHeight;
    dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function themeColors() {
    const dark = document.documentElement.dataset.theme === 'dark';
    return dark
      ? { base: [41, 151, 255], far: 0.11, near: 0.40, glow: 0.10 }
      : { base: [29, 78, 216], far: 0.075, near: 0.30, glow: 0.055 };
  }

  function fieldVector(x, y, time) {
    // Smooth synthetic flow: a weak oscillatory background field plus a local mouse vortex.
    const sx = (x - width * 0.5) / Math.max(width, 1);
    const sy = (y - height * 0.42) / Math.max(height, 1);

    let vx = 0.72 * Math.cos(5.2 * sy + time * 0.00022)
           + 0.38 * Math.sin(4.6 * sx - time * 0.00016);
    let vy = 0.62 * Math.sin(4.4 * sx + time * 0.00018)
           - 0.32 * Math.cos(5.0 * sy);

    if (mouse.active && finePointer) {
      const dx = x - mouse.x;
      const dy = y - mouse.y;
      const dist = Math.hypot(dx, dy) || 1;
      const radius = 230;
      const influence = Math.exp(-(dist * dist) / (2 * radius * radius));

      // Tangential component makes the pointer behave like a small vortex/perturbation.
      const tx = -dy / dist;
      const ty = dx / dist;
      vx += tx * influence * 2.35;
      vy += ty * influence * 2.35;

      // Small radial term prevents the motion from looking perfectly circular.
      vx += (dx / dist) * influence * 0.22;
      vy += (dy / dist) * influence * 0.22;
    }

    if (impulse > 0.001) {
      const dx = x - impulseX;
      const dy = y - impulseY;
      const dist = Math.hypot(dx, dy) || 1;
      const ring = Math.exp(-Math.pow((dist - (1 - impulse) * 290) / 75, 2));
      vx += (dx / dist) * ring * impulse * 1.45;
      vy += (dy / dist) * ring * impulse * 1.45;
    }

    return [vx, vy];
  }

  function drawArrow(x, y, vx, vy, alpha, color) {
    const mag = Math.hypot(vx, vy) || 1;
    const ux = vx / mag;
    const uy = vy / mag;
    const length = 8 + Math.min(mag, 2.6) * 4.7;
    const half = length * 0.46;

    const x1 = x - ux * half;
    const y1 = y - uy * half;
    const x2 = x + ux * half;
    const y2 = y + uy * half;

    ctx.strokeStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${alpha})`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);

    const head = 3.3;
    const px = -uy;
    const py = ux;
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - ux * head - px * head * 0.72, y2 - uy * head - py * head * 0.72);
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - ux * head + px * head * 0.72, y2 - uy * head + py * head * 0.72);
    ctx.stroke();
  }

  function draw(time) {
    if (!visible) return;

    ctx.clearRect(0, 0, width, height);
    const palette = themeColors();

    // Keep the pattern airy so it reads as a computational field, not decoration.
    const spacing = width < 700 ? 74 : 62;
    const top = 92;
    const bottom = Math.min(height, 790);

    for (let y = top; y < bottom; y += spacing) {
      for (let x = 22; x < width; x += spacing) {
        const [vx, vy] = fieldVector(x, y, time);
        let proximity = 0;
        if (mouse.active && finePointer) {
          const d = Math.hypot(x - mouse.x, y - mouse.y);
          proximity = Math.max(0, 1 - d / 285);
        }
        const alpha = palette.far + proximity * (palette.near - palette.far);
        drawArrow(x, y, vx, vy, alpha, palette.base);
      }
    }

    if (mouse.active && finePointer) {
      const radius = 92 + 16 * Math.sin(time * 0.0032);
      const gradient = ctx.createRadialGradient(mouse.x, mouse.y, 0, mouse.x, mouse.y, radius);
      gradient.addColorStop(0, `rgba(${palette.base[0]}, ${palette.base[1]}, ${palette.base[2]}, ${palette.glow})`);
      gradient.addColorStop(1, `rgba(${palette.base[0]}, ${palette.base[1]}, ${palette.base[2]}, 0)`);
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(mouse.x, mouse.y, radius, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function frame(time) {
    const dt = Math.min(32, time - lastTime);
    lastTime = time;

    mouse.x += (mouse.tx - mouse.x) * Math.min(1, dt * 0.012);
    mouse.y += (mouse.ty - mouse.y) * Math.min(1, dt * 0.012);
    impulse *= 0.965;

    draw(time);
    if (!reducedMotion) raf = requestAnimationFrame(frame);
  }

  if (finePointer) {
    window.addEventListener('pointermove', (event) => {
      if (event.pointerType && event.pointerType !== 'mouse' && event.pointerType !== 'pen') return;
      mouse.tx = event.clientX;
      mouse.ty = event.clientY;
      mouse.active = true;
    }, { passive: true });

    window.addEventListener('pointerleave', () => {
      mouse.active = false;
    });

    window.addEventListener('pointerdown', (event) => {
      if (event.button !== 0) return;
      impulseX = event.clientX;
      impulseY = event.clientY;
      impulse = 1;
    }, { passive: true });
  }

  document.addEventListener('visibilitychange', () => {
    visible = !document.hidden;
    if (visible && !reducedMotion && !raf) {
      lastTime = performance.now();
      raf = requestAnimationFrame(frame);
    } else if (!visible && raf) {
      cancelAnimationFrame(raf);
      raf = 0;
    }
  });

  window.addEventListener('resize', () => {
    resize();
    if (reducedMotion) draw(performance.now());
  }, { passive: true });

  resize();
  if (reducedMotion) {
    draw(performance.now());
  } else {
    raf = requestAnimationFrame(frame);
  }
})();
