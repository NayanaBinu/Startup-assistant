/* novaux nav.js – Shared navigation behaviours */
(function () {
  'use strict';

  /* ── Scroll-reactive navbar ── */
  const nav = document.getElementById('mainNav');
  if (nav) {
    const onScroll = () => {
      nav.classList.toggle('scrolled', window.scrollY > 20);
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  /* ── Hamburger / mobile drawer ── */
  const hamburger = document.getElementById('navHamburger');
  const drawer    = document.getElementById('mobileDrawer');
  if (hamburger && drawer) {
    hamburger.addEventListener('click', () => {
      const open = hamburger.classList.toggle('open');
      drawer.classList.toggle('open', open);
      // Prevent body scroll when drawer open
      document.body.style.overflow = open ? 'hidden' : '';
    });
    // Close drawer on link click
    drawer.querySelectorAll('a').forEach(a => {
      a.addEventListener('click', () => {
        hamburger.classList.remove('open');
        drawer.classList.remove('open');
        document.body.style.overflow = '';
      });
    });
    // Close on outside click
    document.addEventListener('click', (e) => {
      if (!nav.contains(e.target) && !drawer.contains(e.target)) {
        hamburger.classList.remove('open');
        drawer.classList.remove('open');
        document.body.style.overflow = '';
      }
    });
  }

  /* ── Active link highlight ── */
  const path = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.novaux-links li a, .nav-mobile-drawer ul li a').forEach(a => {
    const href = a.getAttribute('href') || '';
    if (href === path || (path === '' && href === 'index.html')) {
      a.classList.add('nav-active');
    }
  });
})();
