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
  const hasReference = section.hasAttribute('data-reference-review');
  const compareButtons = section.querySelectorAll('[data-compare]');
  let camera = hasReference ? 1 : -1;
  let comparison = 'all';
  const streamCount = hasReference ? 3 : 2;
  const streamPairs = { generators: [1, 2], 'reference-open': [0, 1], 'reference-peer': [0, 2] };
  const showTriptych = () => hasReference && comparison === 'all';
  const modes = section.querySelector('.camera-modes:not(.reference-modes)');
  const viewButtons = modes.querySelectorAll('[data-view]');
  const names = [...viewButtons].slice(1).map(button => button.textContent);
  const labels = getComputedStyle(stage);
  const divider = labels.getPropertyValue('--white').trim();
  const surface = labels.getPropertyValue('--glass').trim();
  const caption = section.querySelector('.clip-caption');
  const components = section.querySelector('.qtes-components');
  let activeChoice = section.querySelector('.clip-choice[aria-pressed=true]');

  function updateView() {
    const all = camera === -1;
    const triple = showTriptych();
    canvas.width = all || triple ? 960 : 640;
    canvas.height = triple ? (all ? 120 : 180) : 360;
    stage.classList.toggle('three-streams', triple);
    slider.hidden = !stage.classList.contains('interactive') || triple;
    compareButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.compare === comparison)));
    stage.classList.toggle('all-views', all);
    caption.textContent = activeChoice.dataset.caption.replace(
      /^(.*?) \/ .*? \/ (.*?)$/, (_, scene, length) =>
        scene
        + ' / ' + (all ? 'All six cameras' : names[camera]) + ' / ' + length
    );
    viewButtons.forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.view) === camera)));
    draw();
  }

  function draw() {
    const frame = videoReady ? video : poster;
    const width = videoReady ? video.videoWidth : poster.naturalWidth;
    const height = videoReady ? video.videoHeight : poster.naturalHeight;
    if (!width || !height || (!videoReady && !poster.complete)) return;
    const streamWidth = width / streamCount;
    const tileWidth = streamWidth / 3;
    const tileHeight = height / 2;
    const split = Number(slider.value) / 100;
    const all = camera === -1;
    const triple = showTriptych();
    const paneWidth = triple ? canvas.width / 3 : canvas.width;
    const destinationWidth = all ? paneWidth / 3 : paneWidth;
    const destinationHeight = all ? canvas.height / 2 : canvas.height;
    const pair = hasReference ? streamPairs[comparison] : [0, 1];

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
      context.strokeStyle = surface;
      context.lineWidth = 2;
      context.strokeRect(x, y, destinationWidth, destinationHeight);
      context.fillStyle = divider;
      if (!triple && split > 0 && split < 1) context.fillRect(x + cut - 1, y, 2, destinationHeight);
      if (!all) return;
      context.font = triple ? '600 10px system-ui' : '600 15px system-ui';
      const name = names[index];
      const labelWidth = context.measureText(name).width;
      const metrics = triple ? [5, 19, 10, 14, 10, 9] : [7, 27, 12, 21, 13, 11];
      const [boxX, boxY, padding, boxHeight, textX, textY] = metrics;
      context.fillStyle = surface;
      context.fillRect(x + boxX, y + destinationHeight - boxY, labelWidth + padding, boxHeight);
      context.fillStyle = divider;
      context.fillText(name, x + textX, y + destinationHeight - textY);
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
    canvas.hidden = toggle.hidden = modes.hidden = false;
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
    toggle.textContent = 'Pause';
    cancelAnimationFrame(nextFrame);
    animate();
  });
  video.addEventListener('pause', () => {
    toggle.textContent = 'Play';
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
    section.querySelector('.model-left').textContent = left;
    section.querySelector('.model-right').textContent = right;
    section.querySelector('.score-left').textContent = leftScore;
    section.querySelector('.score-right').textContent = rightScore;
    if (components) components.textContent = choice.dataset.components;
    if (hasReference) {
      section.querySelector('.score-reference').textContent = choice.dataset.referenceScore;
      section.querySelector('.score-gap').textContent = choice.dataset.gap;
    }
    video.setAttribute('aria-label', hasReference
      ? 'Recorded reference, ' + left + ' and ' + right + ' matched clips'
      : left + ' and ' + right + ' paired replay');
    updateView();
    if (play && !reduceMotion.matches) video.play().catch(() => {});
  }

  choices.forEach(choice => choice.addEventListener('click', () => chooseClip(choice, true)));
});
