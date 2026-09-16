(() => {
  const root = document.documentElement;

  // ---------- Theme: explicit Light / Dark choices ----------
  let savedTheme = 'light';
  try {
    savedTheme = localStorage.getItem('ronit-theme') || 'light';
  } catch (_) {}
  if (!['light', 'dark'].includes(savedTheme)) savedTheme = 'light';
  root.dataset.theme = savedTheme;

  const themeStyles = document.createElement('style');
  themeStyles.id = 'theme-styles';
  themeStyles.textContent = `
    html[data-theme="light"] { color-scheme: light; }

    html[data-theme="dark"] {
      color-scheme: dark;
      --bg: #000000;
      --paper: #1d1d1f;
      --ink: #f5f5f7;
      --muted: #a1a1a6;
      --line: #2c2c2e;
      --accent: #2997ff;
      --accent-soft: #0b1b2b;
      --deep: #0a84ff;
      --chip: #2c2c2e;
      --shadow: 0 18px 60px rgba(0,0,0,.48);
    }

    body,
    .site-header,
    .stack-card,
    .project-card,
    .contact-card,
    .cv-shell,
    .social-link,
    .chip,
    .news-empty,
    .photo-frame,
    .photo-note,
    .project-visual,
    .project-tags span,
    .theme-switcher,
    .theme-choice {
      transition: background-color .42s ease, color .42s ease, border-color .42s ease, box-shadow .42s ease;
    }

    html[data-theme="dark"] body { background: #000; color: #f5f5f7; }
    html[data-theme="dark"] .site-header {
      background: rgba(0,0,0,.76);
      border-bottom-color: rgba(255,255,255,.12);
      -webkit-backdrop-filter: saturate(180%) blur(20px);
      backdrop-filter: saturate(180%) blur(20px);
    }
    html[data-theme="dark"] .brand { color: #f5f5f7; }
    html[data-theme="dark"] .nav-links a { color: #a1a1a6; }
    html[data-theme="dark"] .nav-links a:hover,
    html[data-theme="dark"] .nav-links a.active { color: #f5f5f7; border-color: #2997ff; }

    html[data-theme="dark"] .hero-kicker,
    html[data-theme="dark"] .hero-copy,
    html[data-theme="dark"] .section-intro,
    html[data-theme="dark"] .page-hero p,
    html[data-theme="dark"] .project-body p,
    html[data-theme="dark"] .pub-meta,
    html[data-theme="dark"] .about-intro,
    html[data-theme="dark"] .timeline-item .meta,
    html[data-theme="dark"] .contact-line span,
    html[data-theme="dark"] .footer { color: #a1a1a6; }

    html[data-theme="dark"] .timeline-item .role { color: #d2d2d7; }
    html[data-theme="dark"] .stack-card h3,
    html[data-theme="dark"] .pub-year { color: #a1a1a6; }

    html[data-theme="dark"] .social-link,
    html[data-theme="dark"] .stack-card,
    html[data-theme="dark"] .project-card,
    html[data-theme="dark"] .contact-card,
    html[data-theme="dark"] .cv-shell {
      background: #1d1d1f;
      border-color: #2c2c2e;
    }
    html[data-theme="dark"] .social-link:hover {
      border-color: #48484a;
      box-shadow: 0 8px 28px rgba(0,0,0,.35);
    }

    html[data-theme="dark"] .chip {
      background: #2c2c2e;
      border-color: #3a3a3c;
      color: #f5f5f7;
    }

    html[data-theme="dark"] .news-empty {
      background: linear-gradient(135deg, #1d1d1f, #111113);
      border-color: #2c2c2e;
      color: #a1a1a6;
    }
    html[data-theme="dark"] .news-empty strong { color: #f5f5f7; }

    html[data-theme="dark"] .photo-frame {
      background: #1d1d1f;
      border-color: #2c2c2e;
      box-shadow: 0 22px 70px rgba(0,0,0,.55);
    }
    html[data-theme="dark"] .photo-note {
      background: rgba(29,29,31,.94);
      border: 1px solid #3a3a3c;
      box-shadow: 0 16px 44px rgba(0,0,0,.48);
    }
    html[data-theme="dark"] .photo-note strong { color: #64a8ff; }

    html[data-theme="dark"] .project-visual {
      background:
        radial-gradient(circle at 22% 34%, rgba(41,151,255,.20), transparent 26%),
        radial-gradient(circle at 78% 62%, rgba(90,200,250,.14), transparent 28%),
        linear-gradient(135deg, #15171a, #08090b);
    }
    html[data-theme="dark"] .project-visual::after {
      background-image: repeating-radial-gradient(ellipse at center, transparent 0 18px, rgba(41,151,255,.10) 19px 20px);
    }
    html[data-theme="dark"] .project-number { color: #64a8ff; }
    html[data-theme="dark"] .project-symbol { color: #f5f5f7; }
    html[data-theme="dark"] .project-tags span {
      color: #d2d2d7;
      background: #2c2c2e;
    }

    html[data-theme="dark"] .timeline { border-left-color: #3a3a3c; }
    html[data-theme="dark"] .timeline-item::before { background: #000; }
    html[data-theme="dark"] .button {
      background: #f5f5f7;
      color: #000;
    }
    html[data-theme="dark"] .button.secondary {
      background: #2c2c2e;
      color: #f5f5f7;
    }

    /* Floating top-right theme control, styled like an Apple control capsule. */
    .theme-switcher {
      position: fixed;
      top: 88px;
      right: 20px;
      z-index: 100;
      display: inline-flex;
      gap: 3px;
      padding: 4px;
      border-radius: 999px;
      background: rgba(255,255,255,.82);
      border: 1px solid rgba(17,24,39,.10);
      box-shadow: 0 8px 28px rgba(0,0,0,.10);
      -webkit-backdrop-filter: saturate(180%) blur(20px);
      backdrop-filter: saturate(180%) blur(20px);
    }
    html[data-theme="dark"] .theme-switcher {
      background: rgba(29,29,31,.78);
      border-color: rgba(255,255,255,.13);
      box-shadow: 0 8px 32px rgba(0,0,0,.40);
    }
    .theme-choice {
      appearance: none;
      border: 0;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 8px 11px;
      border-radius: 999px;
      background: transparent;
      color: #6e6e73;
      font: inherit;
      font-size: .78rem;
      font-weight: 680;
      line-height: 1;
    }
    .theme-choice svg { width: 14px; height: 14px; display: block; }
    .theme-choice:hover { color: #1d1d1f; }
    .theme-choice[aria-pressed="true"] {
      background: #ffffff;
      color: #1d1d1f;
      box-shadow: 0 1px 5px rgba(0,0,0,.12);
    }
    html[data-theme="dark"] .theme-choice { color: #a1a1a6; }
    html[data-theme="dark"] .theme-choice:hover { color: #f5f5f7; }
    html[data-theme="dark"] .theme-choice[aria-pressed="true"] {
      background: #3a3a3c;
      color: #f5f5f7;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,.06), 0 1px 5px rgba(0,0,0,.30);
    }

    @media (max-width: 900px) {
      .theme-switcher { top: 126px; right: 14px; }
    }
    @media (max-width: 600px) {
      .theme-switcher { top: 132px; right: 10px; }
      .theme-choice { padding: 8px 9px; }
      .theme-choice span { display: none; }
    }
    @media (prefers-reduced-motion: reduce) {
      body, .site-header, .stack-card, .project-card, .contact-card, .cv-shell,
      .social-link, .chip, .news-empty, .photo-frame, .photo-note, .project-visual,
      .project-tags span, .theme-switcher, .theme-choice { transition: none !important; }
    }
  `;
  document.head.appendChild(themeStyles);

  const sunIcon = `
    <svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round">
      <circle cx="12" cy="12" r="3.6"></circle>
      <path d="M12 2.3v2.1M12 19.6v2.1M4.4 4.4l1.5 1.5M18.1 18.1l1.5 1.5M2.3 12h2.1M19.6 12h2.1M4.4 19.6l1.5-1.5M18.1 5.9l1.5-1.5"></path>
    </svg>`;
  const moonIcon = `
    <svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
      <path d="M20.2 15.2A8.2 8.2 0 0 1 8.8 3.8 8.5 8.5 0 1 0 20.2 15.2Z"></path>
    </svg>`;

  function setTheme(theme) {
    root.dataset.theme = theme;
    try { localStorage.setItem('ronit-theme', theme); } catch (_) {}
    document.querySelectorAll('.theme-choice').forEach((button) => {
      button.setAttribute('aria-pressed', button.dataset.themeChoice === theme ? 'true' : 'false');
    });
  }

  function buildThemeSwitcher() {
    if (document.querySelector('.theme-switcher')) return;

    const switcher = document.createElement('div');
    switcher.className = 'theme-switcher';
    switcher.setAttribute('role', 'group');
    switcher.setAttribute('aria-label', 'Color theme');
    switcher.innerHTML = `
      <button class="theme-choice" type="button" data-theme-choice="light" aria-label="Use light mode">
        ${sunIcon}<span>Light</span>
      </button>
      <button class="theme-choice" type="button" data-theme-choice="dark" aria-label="Use dark mode">
        ${moonIcon}<span>Dark</span>
      </button>`;

    switcher.addEventListener('click', (event) => {
      const button = event.target.closest('.theme-choice');
      if (!button) return;
      setTheme(button.dataset.themeChoice);
    });

    document.body.appendChild(switcher);
    setTheme(root.dataset.theme || 'light');
  }

  // ---------- Existing flow-in animation ----------
  const selector = [
    '.hero > div:first-child',
    '.hero .photo-wrap',
    '.page-hero > *',
    '.section-head',
    '.stack-card',
    '.news-empty',
    '.project-card',
    '.pub-group',
    '.about-grid > div',
    '.timeline-section',
    '.cv-shell',
    '.contact-card'
  ].join(',');

  function runFlowAnimation() {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

    const items = Array.from(document.querySelectorAll(selector));

    items.forEach((el, index) => {
      el.style.animation = 'none';
      el.getAnimations().forEach(animation => animation.cancel());

      const isPhoto = el.classList.contains('photo-wrap');
      const distance = isPhoto ? 'translate3d(38px, 0, 0)' : 'translate3d(0, 28px, 0)';
      const delay = 70 + Math.min(index * 65, 650);

      el.animate(
        [
          { opacity: 0, transform: distance, filter: 'blur(5px)' },
          { opacity: 1, transform: 'translate3d(0, 0, 0)', filter: 'blur(0px)' }
        ],
        {
          duration: 950,
          delay,
          easing: 'cubic-bezier(.22,.72,.2,1)',
          fill: 'both'
        }
      );
    });
  }

  buildThemeSwitcher();

  // pageshow fires on normal navigation and when a page is restored from browser cache.
  window.addEventListener('pageshow', () => {
    setTheme(root.dataset.theme || savedTheme);
    requestAnimationFrame(() => {
      requestAnimationFrame(runFlowAnimation);
    });
  });
})();
