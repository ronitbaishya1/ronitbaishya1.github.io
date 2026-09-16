(() => {
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
      // Disable the older CSS keyframe animation so this version is reliable.
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

  // pageshow fires on normal navigation and when a page is restored from browser cache.
  window.addEventListener('pageshow', () => {
    requestAnimationFrame(() => {
      requestAnimationFrame(runFlowAnimation);
    });
  });
})();
