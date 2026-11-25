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
  let nextFrame = 0;
  let videoReady = false;
  let camera = -1;
  const modes = section.querySelector('.camera-modes');
  const viewButtons = modes.querySelectorAll('[data-view]');
  const names = [...viewButtons].slice(1).map(button => button.textContent);
  const labels = getComputedStyle(stage);
  const divider = labels.getPropertyValue('--white').trim();
  const surface = labels.getPropertyValue('--glass').trim();
  const caption = section.querySelector('.clip-caption');
  let activeChoice = section.querySelector('.clip-choice[aria-pressed=true]');

  function updateView() {
    const all = camera === -1;
    canvas.width = all ? 960 : 640;
    canvas.height = 360;
    stage.classList.toggle('all-views', all);
    caption.textContent = activeChoice.dataset.caption.replace(
      /^(.*?) \/ .*? \/ (.*?)$/, (_, scene, length) =>
        scene + ' / ' + (all ? 'All six cameras' : names[camera]) + ' / ' + length
    );
    viewButtons.forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.view) === camera)));
    draw();
  }

  function draw() {
    const frame = videoReady ? video : poster;
    const width = videoReady ? video.videoWidth : poster.naturalWidth;
    const height = videoReady ? video.videoHeight : poster.naturalHeight;
    if (!width || !height || (!videoReady && !poster.complete)) return;
    const tileWidth = width / 6;
    const tileHeight = height / 2;
    const split = Number(slider.value) / 100;
    const all = camera === -1;
    const destinationWidth = all ? canvas.width / 3 : canvas.width;
    const destinationHeight = all ? canvas.height / 2 : canvas.height;

    function tile(index) {
      const column = index % 3;
      const row = Math.floor(index / 3);
      const x = all ? column * destinationWidth : 0;
      const y = all ? row * destinationHeight : 0;
      const sx = column * tileWidth;
      const sy = row * tileHeight;
      const cut = destinationWidth * split;
      if (split > 0) {
        context.drawImage(frame, sx, sy, tileWidth * split, tileHeight, x, y, cut, destinationHeight);
      }
      if (split < 1) {
        context.drawImage(frame, width / 2 + sx + tileWidth * split, sy, tileWidth * (1 - split), tileHeight,
          x + cut, y, destinationWidth - cut, destinationHeight);
      }
      if (!all) return;
      context.strokeStyle = surface;
      context.lineWidth = 2;
      context.strokeRect(x, y, destinationWidth, destinationHeight);
      context.fillStyle = divider;
      if (split > 0 && split < 1) context.fillRect(x + cut - 1, y, 2, destinationHeight);
      context.font = '600 15px system-ui';
      const name = names[index];
      const labelWidth = context.measureText(name).width;
      context.fillStyle = surface;
      context.fillRect(x + 7, y + destinationHeight - 27, labelWidth + 12, 21);
      context.fillStyle = divider;
      context.fillText(name, x + 13, y + destinationHeight - 11);
    }

    if (all) names.forEach((_, index) => tile(index));
    else tile(camera);
  }

  function animate() {
    draw();
    if (!video.paused) nextFrame = requestAnimationFrame(animate);
  }

  poster.addEventListener('load', () => {
    updateView();
    stage.classList.add('interactive');
    canvas.hidden = slider.hidden = toggle.hidden = modes.hidden = false;
    video.controls = false;
  });
  poster.src = video.poster;
  viewButtons.forEach(button => button.addEventListener('click', () => {
    camera = Number(button.dataset.view);
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
    activeChoice = choice;
    const {clip, left, right, leftScore, rightScore} = choice.dataset;
    const wasPlaying = !video.paused;
    video.pause();
    videoReady = false;
    video.src = 'media/atlas-six/' + clip + '.mp4';
    video.poster = poster.src = 'media/atlas-six/' + clip + '.jpg';
    choices.forEach(button => button.setAttribute('aria-pressed', String(button === choice)));
    section.querySelector('.model-left').textContent = left;
    section.querySelector('.model-right').textContent = right;
    section.querySelector('.score-left').textContent = leftScore;
    section.querySelector('.score-right').textContent = rightScore;
    updateView();
    if (wasPlaying || !reduceMotion.matches) video.play().catch(() => {});
  }));
});
