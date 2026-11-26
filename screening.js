// Keep fixed-header clearance correct when links wrap or fonts finish loading.
(() => {
  const navigation = document.querySelector('.cover-nav');
  if (!navigation) return;
  const update = () => {
    const height = `${navigation.getBoundingClientRect().height}px`;
    if (document.documentElement.style.getPropertyValue('--page-nav-height') === height) return;
    document.documentElement.style.setProperty('--page-nav-height', height);
    dispatchEvent(new CustomEvent('page-nav-resize'));
  };
  if ('ResizeObserver' in window) new ResizeObserver(update).observe(navigation);
  addEventListener('resize', update, { passive: true });
  document.fonts?.ready.then(update);
  update();
})();

const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
const cover = document.querySelector('#cover-film');
let coverWantsPlayback = !reduceMotion.matches;
let coverVisible = true;
cover.autoplay = coverWantsPlayback;

// Suspend in hidden views or during body playback; resume on returning to cover.
function updateCoverPlayback() {
  if (coverWantsPlayback && coverVisible && !document.hidden) cover.play().catch(() => {});
  else cover.pause();
}
document.addEventListener('play', event => {
  if (event.target instanceof HTMLMediaElement && event.target !== cover) {
    coverWantsPlayback = false;
    cover.pause();
  }
}, true);
document.addEventListener('visibilitychange', updateCoverPlayback);
reduceMotion.addEventListener('change', () => {
  coverWantsPlayback = !reduceMotion.matches;
  cover.autoplay = coverWantsPlayback;
  updateCoverPlayback();
});
if ('IntersectionObserver' in window) {
  new IntersectionObserver(([entry]) => {
    const wasVisible = coverVisible;
    coverVisible = entry.isIntersecting && entry.intersectionRatio > 0;
    if (coverVisible && !wasVisible) coverWantsPlayback = !reduceMotion.matches;
    updateCoverPlayback();
  }).observe(cover.closest('.cover-pin'));
}
updateCoverPlayback();

document.querySelectorAll('.visual-metric').forEach(section => {
  const stage = section.querySelector('.motion-stage');
  const video = stage.querySelector('video');
  const canvas = stage.querySelector('canvas');
  const context = canvas.getContext('2d');
  if (!context) return;

  const slider = stage.querySelector('input');
  const toggle = section.querySelector('.motion-toggle');
  const choices = section.querySelectorAll('.clip-choice');
  const poster = new Image();
  let nextFrame = 0;
  let videoReady = false;
  const hasReference = section.hasAttribute('data-reference-review');
  const compareButtons = section.querySelectorAll('[data-compare]');
  let camera = 1;
  let comparison = hasReference ? 'generators' : 'all';
  const streamCount = hasReference ? 3 : 2;
  const streamPairs = { generators: [1, 2], 'reference-open': [0, 1], 'reference-peer': [0, 2] };
  const showTriptych = () => hasReference && comparison === 'all';
  const modes = section.querySelector('.camera-modes:not(.reference-modes)');
  const viewButtons = modes.querySelectorAll('[data-view]');
  const getNames = () => [...viewButtons].map(button => button.textContent.trim());
  const isZh = () => document.documentElement.lang === 'zh-CN';
  const playText = paused => (isZh() ? (paused ? '▶ 播放' : '❚❚ 暂停') : (paused ? '▶ Play' : '❚❚ Pause'));
  const localizeStream = label => {
    if (!isZh()) return label;
    if (label === 'Good example') return '优秀样例';
    if (label === 'Bad example') return '较差样例';
    return label;
  };
  const labels = getComputedStyle(stage);
  const divider = labels.getPropertyValue('--white').trim();
  const surface = labels.getPropertyValue('--glass').trim();
  const caption = section.querySelector('.clip-caption');
  const components = section.querySelector('.qtes-components');
  const badgeRef = section.querySelector('.score-stream-reference');
  const badgeLeft = section.querySelector('.score-stream-open');
  const badgeRight = section.querySelector('.score-stream-peer');
  let activeChoice = section.querySelector('.clip-choice[aria-pressed=true]');

  function updateView() {
    const names = getNames();
    const triple = showTriptych();
    const all = hasReference ? !triple : camera === -1;
    canvas.width = 1200;
    canvas.height = triple ? 225 : 450;
    stage.classList.toggle('three-streams', triple);
    slider.hidden = !stage.classList.contains('interactive') || triple;
    modes.hidden = !stage.classList.contains('interactive') || (hasReference && !triple);
    compareButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.compare === comparison)));
    stage.classList.toggle('all-views', all);
    if (hasReference && badgeRef && badgeLeft && badgeRight) {
      badgeRef.hidden = comparison === 'generators';
      badgeLeft.hidden = comparison === 'reference-peer';
      badgeRight.hidden = comparison === 'reference-open';
    }
    toggle.textContent = playText(video.paused);
    if (caption && activeChoice.dataset.caption) {
      caption.textContent = activeChoice.dataset.caption.replace(
        /^(.*?) \/ .*? \/ (.*?)$/, (_, scene, length) =>
          scene
          + ' / ' + (all ? 'All six cameras' : names[camera]) + ' / ' + length
      );
    }
    viewButtons.forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.view) === camera)));
    draw();
  }

  function draw() {
    const names = getNames();
    const frame = videoReady ? video : poster;
    const width = videoReady ? video.videoWidth : poster.naturalWidth;
    const height = videoReady ? video.videoHeight : poster.naturalHeight;
    if (!width || !height || (!videoReady && !poster.complete)) return;
    const streamWidth = width / streamCount;
    const tileWidth = streamWidth / 3;
    const tileHeight = height / 2;
    const split = Number(slider.value) / 100;
    const triple = showTriptych();
    const all = hasReference ? !triple : camera === -1;
    const paneWidth = triple ? canvas.width / 3 : canvas.width;
    const destinationWidth = all ? paneWidth / 3 : paneWidth;
    const destinationHeight = all ? canvas.height / 2 : canvas.height;
    const pair = hasReference ? streamPairs[comparison] : [0, 1];

    function drawRoundedBox(rx, ry, rw, rh, radius) {
      context.beginPath();
      if (typeof context.roundRect === 'function') context.roundRect(rx, ry, rw, rh, radius);
      else context.rect(rx, ry, rw, rh);
    }

    function paintTile(index, pane = 0) {
      const column = index % 3;
      const row = Math.floor(index / 3);
      const x = pane * paneWidth + (all ? column * destinationWidth : 0);
      const y = all ? row * destinationHeight : 0;
      const sx = column * tileWidth;
      const sy = row * tileHeight;
      const cut = destinationWidth * split;
      if (triple) {
        context.drawImage(frame, pane * streamWidth + sx, sy, tileWidth, tileHeight,
          x, y, destinationWidth, destinationHeight);
      } else {
        if (split > 0) {
          context.drawImage(frame, pair[0] * streamWidth + sx, sy, tileWidth * split, tileHeight,
            x, y, cut, destinationHeight);
        }
        if (split < 1) {
          context.drawImage(frame, pair[1] * streamWidth + sx + tileWidth * split, sy,
            tileWidth * (1 - split), tileHeight, x + cut, y, destinationWidth - cut, destinationHeight);
        }
      }
      if (!all && !triple) return;
      context.strokeStyle = 'rgba(15, 23, 42, 0.75)';
      context.lineWidth = 2;
      context.strokeRect(x, y, destinationWidth, destinationHeight);
      if (!triple && split > 0 && split < 1) {
        context.fillStyle = '#ffffff';
        context.fillRect(x + cut - 1, y, 2, destinationHeight);
        if (split > 0.1 && split < 0.9) {
          const handleW = 28;
          const handleH = 18;
          const hx = x + cut - handleW / 2;
          const hy = y + destinationHeight / 2 - handleH / 2;
          context.save();
          context.fillStyle = 'rgba(15, 23, 42, 0.76)';
          drawRoundedBox(hx, hy, handleW, handleH, 9);
          context.fill();
          context.strokeStyle = 'rgba(255, 255, 255, 0.32)';
          context.lineWidth = 1;
          context.stroke();
          context.fillStyle = '#ffffff';
          context.font = '600 12px system-ui, sans-serif';
          context.textAlign = 'center';
          context.textBaseline = 'middle';
          context.fillText('‹ ›', x + cut, y + destinationHeight / 2 - 0.5);
          context.restore();
        }
      }
      if (!all) return;
      context.save();
      context.font = '600 15px "Google Sans", "Noto Sans", "Noto Sans SC", system-ui, sans-serif';
      const name = names[index];
      const labelWidth = context.measureText(name).width;
      const boxW = labelWidth + 22;
      const boxH = 28;
      context.fillStyle = 'rgba(15, 23, 42, 0.76)';
      drawRoundedBox(x + 10, y + 10, boxW, boxH, 14);
      context.fill();
      context.fillStyle = '#ffffff';
      context.textAlign = 'left';
      context.textBaseline = 'middle';
      context.fillText(name, x + 21, y + 10 + boxH / 2);
      context.restore();
    }

    for (let pane = 0; pane < (triple ? 3 : 1); pane++) {
      if (all) names.forEach((_, index) => paintTile(index, pane));
      else paintTile(camera, pane);
    }
  }

  function animate() {
    draw();
    if (!video.paused) nextFrame = requestAnimationFrame(animate);
  }

  poster.addEventListener('load', () => {
    stage.classList.add('interactive');
    canvas.hidden = toggle.hidden = false;
    if (hasReference) section.querySelector('.reference-modes').hidden = false;
    updateView();
    video.controls = false;
  });
  poster.src = video.poster;
  viewButtons.forEach(button => button.addEventListener('click', () => {
    camera = Number(button.dataset.view);
    updateView();
  }));
  compareButtons.forEach(button => button.addEventListener('click', () => {
    comparison = button.dataset.compare;
    updateView();
  }));
  slider.addEventListener('input', () => {
    stage.style.setProperty('--cut', slider.value + '%');
    draw();
  });
  toggle.addEventListener('click', () => {
    if (video.paused) video.play().catch(() => {});
    else video.pause();
  });
  video.addEventListener('loadeddata', () => {
    videoReady = true;
    draw();
  });
  video.addEventListener('play', () => {
    document.querySelectorAll('.visual-metric video, #cover-film').forEach(other => {
      if (other !== video) other.pause();
    });
    toggle.textContent = playText(false);
    cancelAnimationFrame(nextFrame);
    animate();
  });
  video.addEventListener('pause', () => {
    toggle.textContent = playText(true);
    cancelAnimationFrame(nextFrame);
  });

  function chooseClip(choice, play) {
    if (choice.getAttribute('aria-pressed') === 'true') return;
    activeChoice = choice;
    const {clip, left, right, leftScore, rightScore} = choice.dataset;
    video.pause();
    videoReady = false;
    const directory = hasReference ? 'media/quality-reference/' : 'media/atlas-scored/';
    video.src = directory + clip + '.mp4';
    video.poster = poster.src = directory + clip + '.jpg';
    choices.forEach(button => button.setAttribute('aria-pressed', String(button === choice)));
    section.querySelector('.model-left').textContent = localizeStream(left);
    section.querySelector('.model-right').textContent = localizeStream(right);
    section.querySelector('.score-left').textContent = leftScore;
    section.querySelector('.score-right').textContent = rightScore;
    if (components) components.textContent = choice.dataset.components;
    if (hasReference) {
      section.querySelector('.score-reference').textContent = choice.dataset.referenceScore;
      const gap = section.querySelector('.score-gap');
      if (gap && choice.dataset.gap) gap.textContent = choice.dataset.gap;
    }
    video.setAttribute('aria-label', hasReference
      ? 'Recorded reference, ' + left + ' and ' + right + ' matched clips'
      : left + ' and ' + right + ' paired replay');
    updateView();
    if (play && !reduceMotion.matches) video.play().catch(() => {});
  }

  choices.forEach(choice => choice.addEventListener('click', () => chooseClip(choice, true)));
  addEventListener('driveaudit-lang-change', () => {
    toggle.textContent = playText(video.paused);
    draw();
  });
});
