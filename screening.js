const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
const cover = document.querySelector('#cover-film');
const coverToggle = document.querySelector('#cover-toggle');

coverToggle.addEventListener('click', () => {
  if (cover.paused) cover.play().catch(() => {});
  else cover.pause();
});
cover.addEventListener('play', () => { coverToggle.textContent = 'Pause cover'; });
cover.addEventListener('pause', () => { coverToggle.textContent = 'Play cover'; });
if (!reduceMotion.matches) cover.play().catch(() => {});

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
  const scope = section.querySelector('.clip-scope');
  const benchmarkScope = scope.textContent;
  let nextFrame = 0;
  let videoReady = false;

  function draw() {
    const frame = videoReady ? video : poster;
    const width = videoReady ? video.videoWidth : poster.naturalWidth;
    const height = videoReady ? video.videoHeight : poster.naturalHeight;
    const half = width / 2;
    if (!half || !height || (!videoReady && !poster.complete)) return;
    const leftWidth = Math.round(half * Number(slider.value) / 100);
    const rightWidth = half - leftWidth;
    if (leftWidth) context.drawImage(frame, 0, 0, leftWidth, height, 0, 0, leftWidth, height);
    if (rightWidth) context.drawImage(frame, half + leftWidth, 0, rightWidth, height, leftWidth, 0, rightWidth, height);
  }

  function animate() {
    draw();
    if (!video.paused) nextFrame = requestAnimationFrame(animate);
  }

  poster.addEventListener('load', () => {
    draw();
    stage.classList.add('interactive');
    canvas.hidden = slider.hidden = toggle.hidden = false;
    video.controls = false;
  });
  poster.src = video.poster;
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
    toggle.textContent = 'Pause';
    cancelAnimationFrame(nextFrame);
    animate();
  });
  video.addEventListener('pause', () => {
    toggle.textContent = 'Play';
    cancelAnimationFrame(nextFrame);
  });

  choices.forEach(choice => choice.addEventListener('click', () => {
    if (choice.getAttribute('aria-pressed') === 'true') return;
    const {clip, caption, left, right, leftScore, rightScore} = choice.dataset;
    const wasPlaying = !video.paused;
    video.pause();
    videoReady = false;
    video.src = 'media/atlas/' + clip + '.mp4';
    video.poster = poster.src = 'media/atlas/' + clip + '.jpg';
    choices.forEach(button => button.setAttribute('aria-pressed', String(button === choice)));
    section.querySelector('.model-left').textContent = left;
    section.querySelector('.model-right').textContent = right;
    section.querySelector('.score-left').textContent = leftScore;
    section.querySelector('.score-right').textContent = rightScore;
    section.querySelector('.clip-caption').textContent = caption;
    scope.textContent = choice.dataset.scope || benchmarkScope;
    if (wasPlaying || !reduceMotion.matches) video.play().catch(() => {});
  }));
});

const cases = Array.from(document.querySelectorAll('.policy-case'));
const caseLinks = document.querySelectorAll('[data-policy-case]');

function showPolicyCase(id) {
  cases.forEach(figure => { figure.hidden = figure.id !== id; });
  caseLinks.forEach(link => {
    if (link.hash === '#' + id) link.setAttribute('aria-current', 'true');
    else link.removeAttribute('aria-current');
  });
}

if (cases.length) {
  const showFromHash = () => {
    const id = location.hash.slice(1);
    if (cases.some(figure => figure.id === id)) showPolicyCase(id);
  };
  showPolicyCase(cases[0].id);
  showFromHash();
  caseLinks.forEach(link => link.addEventListener('click', () => showPolicyCase(link.hash.slice(1))));
  addEventListener('hashchange', showFromHash);
}
