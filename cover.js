(() => {
  const section = document.querySelector('.cover');
  const stage = section?.querySelector('.cover-pin');
  const video = document.querySelector('#cover-film');
  if (!stage || !video || !CSS.supports('position', 'sticky')) return;

  const media = stage.querySelector('.windshield');
  const intro = stage.querySelector('.cover-title');
  const scrollLink = stage.querySelector('.cover-scroll');
  const heading = stage.querySelector('.cover-expanded-heading');
  const dimensions = stage.querySelector('.cover-dimensions');
  const host = stage.querySelector('.cover-tiles');
  const eligible = matchMedia('(min-width: 960px) and (min-height: 680px) and (prefers-reduced-motion: no-preference)');
  const clamp = value => Math.max(0, Math.min(1, value));
  const ease = (start, end, value) => {
    const p = clamp((value - start) / (end - start));
    return p * p * (3 - 2 * p);
  };
  const mix = (start, end, amount) => start + (end - start) * amount;

  // Source cells refer to the existing five-scene, six-view montage, not new clips.
  // Every panel samples this same video clock; layout and source selection are separate.
  const sources = [[0, 0], [2, 1], [1, 2], [4, 4], [0, 3], [3, 2]];
  const layouts = {
    wide: {
      columns: 12, rows: 5, main: [3, 0, 6, 3],
      panels: [
        [0, 0, 0, 3, 3], [1, 0, 3, 3, 2], [2, 9, 0, 3, 2],
        [3, 9, 2, 3, 3], [4, 3, 3, 3, 2], [5, 6, 3, 3, 2],
      ],
    },
    compact: {
      columns: 10, rows: 2, main: [2, 0, 6, 2],
      panels: [
        [0, 0, 0, 2, 1], [1, 0, 1, 2, 1], [4, 8, 0, 2, 1], [3, 8, 1, 2, 1],
      ],
    },
  };
  const tiles = [];
  for (const [column, row] of sources) {
    const frame = document.createElement('div');
    frame.className = 'cover-tile';
    frame.hidden = true;
    const canvas = document.createElement('canvas');
    let context;
    try { context = canvas.getContext('2d', { alpha: false }); } catch { return; }
    if (!context) return;
    const label = document.createElement('span');
    label.className = 'cover-tile-label';
    const scene = document.createElement('span');
    const view = document.createElement('span');
    scene.textContent = `Scene ${String(row + 1).padStart(2, '0')}`;
    view.textContent = `View ${String(column + 1).padStart(2, '0')}`;
    label.append(scene, view);
    frame.append(canvas, label);
    tiles.push({ frame, canvas, context, source: { x: column / 6, y: row / 5, w: 1 / 6, h: 1 / 5 }, visible: false });
  }
  host.append(...tiles.map(tile => tile.frame));

  let active = false;
  let failed = Boolean(video.error);
  let visible = true;
  let geometry = null;
  let progress = 0;
  let blockedSize = '';
  let layoutFrame = null;
  let needsMeasure = false;
  let videoFrame = null;
  const nativeFrames = 'requestVideoFrameCallback' in video;
  const tabOrder = new WeakMap();
  const nativeInert = 'inert' in HTMLElement.prototype;
  const poster = new Image();

  function interactive(element, enabled) {
    element.toggleAttribute('inert', !enabled);
    element.setAttribute('aria-hidden', String(!enabled));
    if (nativeInert) return;
    const links = [...element.querySelectorAll('a, button')];
    if (element.matches('a, button')) links.push(element);
    links.forEach(link => {
      if (!tabOrder.has(link)) tabOrder.set(link, link.getAttribute('tabindex'));
      const original = tabOrder.get(link);
      if (!enabled) link.setAttribute('tabindex', '-1');
      else if (original === null) link.removeAttribute('tabindex');
      else link.setAttribute('tabindex', original);
    });
  }
  function stopFrames() {
    if (videoFrame === null) return;
    if (nativeFrames) video.cancelVideoFrameCallback(videoFrame);
    else cancelAnimationFrame(videoFrame);
    videoFrame = null;
  }

  // Aspect-preserving center crop, in normalized coordinates of the full source.
  function fitCrop(source, destinationAspect, sourceAspect) {
    const crop = { ...source };
    if (crop.w / crop.h * sourceAspect > destinationAspect) {
      const width = crop.h * destinationAspect / sourceAspect;
      crop.x += (crop.w - width) / 2;
      crop.w = width;
    } else {
      const height = crop.w * sourceAspect / destinationAspect;
      crop.y += (crop.h - height) / 2;
      crop.h = height;
    }
    return crop;
  }
  function paint() {
    if (!active || !visible || document.hidden || progress <= 0.15 || video.readyState < 2) return;
    tiles.forEach(({ canvas, context, crop, visible: shown }) => {
      if (!shown || !crop) return;
      context.drawImage(video, crop.x * video.videoWidth, crop.y * video.videoHeight,
        crop.w * video.videoWidth, crop.h * video.videoHeight, 0, 0, canvas.width, canvas.height);
    });
    section.classList.add('has-crop-frames');
  }
  function requestFrames() {
    if (videoFrame !== null || !active || !visible || document.hidden || video.paused || progress <= 0.15) return;
    const next = () => { videoFrame = null; paint(); requestFrames(); };
    videoFrame = nativeFrames ? video.requestVideoFrameCallback(next) : requestAnimationFrame(next);
  }

  function reset() {
    active = false;
    stopFrames();
    section.classList.remove('is-scrollable', 'is-unfolding', 'has-crop-frames');
    for (const name of ['--cover-copy', '--cover-offset', '--cover-reveal', '--cover-shade', '--cover-scroll', '--cover-surface']) {
      section.style.removeProperty(name);
    }
    media.removeAttribute('style');
    video.removeAttribute('style');
    dimensions.removeAttribute('style');
    heading.hidden = dimensions.hidden = true;
    interactive(intro, true);
    interactive(scrollLink, true);
    interactive(heading, false);
    interactive(dimensions, false);
    tiles.forEach(tile => { tile.visible = false; tile.frame.hidden = true; });
    section.removeAttribute('data-progress');
    section.removeAttribute('data-layout');
    document.documentElement.style.removeProperty('--cover-trim');
    geometry = null;
  }
  function measure() {
    const size = `${innerWidth}x${innerHeight}`;
    if (!eligible.matches || failed || blockedSize === size) { reset(); return; }
    active = true;
    section.classList.add('is-scrollable');
    heading.hidden = dimensions.hidden = false;
    const width = stage.clientWidth;
    const height = stage.clientHeight;
    const gutter = Math.max(24, Math.min(96, width * 0.04));
    const gap = Math.max(18, Math.min(28, width * 0.014));
    const availableWidth = Math.min(2080, width - 2 * gutter);
    const boardLeft = (width - availableWidth) / 2;
    // Measure the navigation at the same width as the entire video board.
    Object.assign(dimensions.style, { left: `${boardLeft}px`, right: 'auto', width: `${availableWidth}px` });
    const top = heading.offsetTop + heading.offsetHeight + 22;
    const navigationGap = 24;
    const availableHeight = height - top - dimensions.offsetHeight - navigationGap - 36;
    if (availableHeight < 260) {
      // Do not invent vertical space or shrink readable text to force a sticky layout.
      blockedSize = size;
      reset();
      return;
    }
    const name = width >= 1280 && availableHeight >= 400 ? 'wide' : 'compact';
    const layout = layouts[name];
    section.dataset.layout = name;
    const boardHeight = Math.min(availableHeight, availableWidth / (name === 'wide' ? 2.35 : 2.8));
    const boardTop = top;
    const dimensionsTop = boardTop + boardHeight + navigationGap;
    Object.assign(dimensions.style, { top: `${dimensionsTop}px`, bottom: 'auto' });
    const usedBottom = dimensionsTop + dimensions.offsetHeight;
    const slack = Math.max(0, Math.round(height - usedBottom - 76));
    document.documentElement.style.setProperty('--cover-trim', `${slack}px`);
    const cellWidth = (availableWidth - gap * (layout.columns - 1)) / layout.columns;
    const cellHeight = (boardHeight - gap * (layout.rows - 1)) / layout.rows;
    const rectangle = ([column, row, columns, rows]) => ({
      x: boardLeft + column * (cellWidth + gap), y: boardTop + row * (cellHeight + gap),
      w: columns * cellWidth + (columns - 1) * gap, h: rows * cellHeight + (rows - 1) * gap,
    });
    const ends = Array(tiles.length).fill(null);
    layout.panels.forEach(([id, ...box]) => { ends[id] = rectangle(box); });
    const main = rectangle(layout.main);
    const aspect = video.videoWidth && video.videoHeight ? video.videoWidth / video.videoHeight : 1536 / 720;
    const posterAspect = poster.naturalWidth && poster.naturalHeight ? poster.naturalWidth / poster.naturalHeight : 1536 / 720;
    geometry = { width, height, aspect, main, ends };
    tiles.forEach((tile, index) => {
      const end = ends[index];
      tile.visible = Boolean(end);
      tile.frame.hidden = !end;
      if (!end) return;
      const scale = Math.min(devicePixelRatio || 1, 2, 512 / end.w, 384 / end.h);
      const pixelsWide = Math.max(1, Math.round(end.w * scale));
      const pixelsHigh = Math.max(1, Math.round(end.h * scale));
      if (tile.canvas.width !== pixelsWide) tile.canvas.width = pixelsWide;
      if (tile.canvas.height !== pixelsHigh) tile.canvas.height = pixelsHigh;
      tile.crop = fitCrop(tile.source, pixelsWide / pixelsHigh, aspect);
      const fallback = fitCrop(tile.source, end.w / end.h, posterAspect);
      tile.frame.style.backgroundSize = `${100 / fallback.w}% ${100 / fallback.h}%`;
      tile.frame.style.backgroundPosition = `${fallback.x / (1 - fallback.w) * 100}% ${fallback.y / (1 - fallback.h) * 100}%`;
    });
  }
  function place(element, rectangle) {
    Object.assign(element.style, {
      left: `${rectangle.x}px`, top: `${rectangle.y}px`,
      width: `${rectangle.w}px`, height: `${rectangle.h}px`,
    });
  }
  function updateLayout() {
    if (!active || !geometry) return;
    const travel = Math.max(1, section.offsetHeight - stage.offsetHeight);
    progress = clamp(-section.getBoundingClientRect().top / travel);
    section.dataset.progress = progress.toFixed(3);
    const motion = ease(0.02, 0.72, progress);
    const introFocused = intro.contains(document.activeElement);
    const scrollFocused = document.activeElement === scrollLink;
    const dimensionsFocused = dimensions.contains(document.activeElement);
    const copy = introFocused ? 1 : 1 - ease(0.03, 0.30, progress);
    const reveal = introFocused ? 0 : dimensionsFocused ? 1 : ease(0.76, 0.92, progress);
    section.style.setProperty('--cover-copy', copy);
    section.style.setProperty('--cover-offset', `${-28 * (1 - copy)}px`);
    section.style.setProperty('--cover-reveal', reveal);
    section.style.setProperty('--cover-scroll', scrollFocused ? 1 : copy);
    section.style.setProperty('--cover-shade', 1 - ease(0.04, 0.65, progress));
    // Theme follows physical scroll, not focus-protected copy/reveal opacity.
    section.style.setProperty('--cover-surface', ease(0.20, 0.76, progress));
    section.classList.toggle('is-unfolding', progress > 0.03);
    interactive(intro, copy > 0.12);
    interactive(scrollLink, scrollFocused || copy > 0.12);
    interactive(heading, reveal > 0.75);
    interactive(dimensions, dimensionsFocused || reveal > 0.75);

    const { width, height, aspect, main, ends } = geometry;
    const start = { x: 0, y: 0, w: width, h: height };
    const primary = {};
    for (const key of ['x', 'y', 'w', 'h']) primary[key] = mix(start[key], main[key], motion);
    place(media, primary);
    media.style.borderRadius = `${6 * motion}px`;
    // Keep an undistorted cover fit throughout: no dark letterbox strips.
    const imageHeight = Math.max(primary.h, primary.w / aspect);
    video.style.width = `${imageHeight * aspect}px`;
    video.style.height = `${imageHeight}px`;
    media.style.backgroundSize = `${imageHeight * aspect}px ${imageHeight}px`;
    tiles.forEach(({ frame, visible: shown }, index) => {
      if (!shown) return;
      const end = ends[index];
      const delay = index * 0.011;
      const spread = ease(0.18 + delay, 0.72 + delay, progress);
      const initial = { x: width / 2 - end.w * 0.4, y: height / 2 - end.h * 0.4, w: end.w * 0.8, h: end.h * 0.8 };
      const rect = {};
      for (const key of ['x', 'y', 'w', 'h']) rect[key] = mix(initial[key], end[key], spread);
      place(frame, rect);
      frame.style.opacity = ease(0.05, 0.65, spread);
    });
    if (progress <= 0.15) stopFrames();
    else { paint(); requestFrames(); }
  }
  function schedule(remeasure = false) {
    needsMeasure = needsMeasure || remeasure;
    if (layoutFrame !== null) return;
    layoutFrame = requestAnimationFrame(() => {
      layoutFrame = null;
      if (needsMeasure) { needsMeasure = false; measure(); }
      updateLayout();
    });
  }
  function refreshSize() { blockedSize = ''; schedule(true); }

  scrollLink.addEventListener('click', event => {
    if (!active || progress > 0.85 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    const top = section.getBoundingClientRect().top + scrollY;
    scrollTo({ top: top + (section.offsetHeight - stage.offsetHeight) * 0.98, behavior: 'smooth' });
  });
  addEventListener('scroll', () => schedule(), { passive: true });
  addEventListener('resize', refreshSize, { passive: true });
  addEventListener('page-nav-resize', refreshSize);
  addEventListener('pageshow', () => schedule(true));
  addEventListener('hashchange', () => schedule(true));
  eligible.addEventListener('change', refreshSize);
  stage.addEventListener('focusin', () => schedule());
  stage.addEventListener('focusout', () => schedule());
  video.addEventListener('loadedmetadata', () => schedule(true));
  video.addEventListener('loadeddata', () => { paint(); requestFrames(); });
  video.addEventListener('seeked', paint);
  video.addEventListener('play', requestFrames);
  video.addEventListener('pause', () => { stopFrames(); paint(); });
  video.addEventListener('error', () => { failed = true; schedule(true); });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) stopFrames();
    else { paint(); requestFrames(); }
  });
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting && entry.intersectionRatio > 0;
      if (!visible) stopFrames();
      else { paint(); requestFrames(); }
    }).observe(stage);
  }
  if ('ResizeObserver' in window) {
    const observer = new ResizeObserver(() => schedule(true));
    observer.observe(heading);
    observer.observe(dimensions);
  }
  poster.addEventListener('load', () => schedule(true));
  poster.src = video.poster;
  document.fonts?.ready.then(refreshSize);
  measure();
  updateLayout();
})();
