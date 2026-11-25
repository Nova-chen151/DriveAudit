(() => {
  const panel = document.querySelector('.camera-inspect');
  const stage = panel.querySelector('.stage');
  const slider = panel.querySelector('#split');
  const button = panel.querySelector('#camera-play');
  const choices = [...panel.querySelectorAll('.choices button')];
  const contexts = ['left', 'right'].map(id => document.getElementById(id).getContext('2d'));
  let images, timer, frame = 0, version = 0, interval = 125;

  function draw() {
    contexts.forEach((ctx, i) => ctx.drawImage(images[i], (frame % 8) * 400,
      Math.floor(frame / 8) * 224, 400, 224, 0, 0, 400, 224));
  }
  function stop() {
    clearInterval(timer);
    timer = null;
    button.textContent = 'Play clips';
  }
  function play() {
    stop();
    if (!images) return;
    button.textContent = 'Pause clips';
    timer = setInterval(() => { frame = (frame + 1) % 80; draw(); }, interval);
  }
  async function select(choice) {
    const token = ++version;
    stop();
    images = null;
    button.disabled = true;
    choices.forEach(item => item.setAttribute('aria-pressed', String(item === choice)));
    const {scene, camera, caption} = choice.dataset;
    panel.querySelector('#caption').textContent = caption;
    try {
      const pair = await Promise.all(['opendwm', 'dreamforge'].map(async model => {
        const image = new Image();
        image.src = 'media/camera/' + model + '-' + scene + '-' + camera + '.jpg';
        await image.decode();
        return image;
      }));
      if (token !== version) return;
      images = pair;
      frame = 0;
      interval = (scene === '000001' ? 10040 : 10280) / 80;
      draw();
      button.disabled = false;
      if (panel.open && !matchMedia('(prefers-reduced-motion: reduce)').matches) play();
    } catch {
      if (token === version) panel.querySelector('#caption').textContent = 'Camera frames could not load. Select a view to retry.';
    }
  }
  choices.forEach(c => c.addEventListener('click', () => select(c)));
  panel.addEventListener('toggle', () => {
    if (panel.open) select(choices.find(c => c.getAttribute('aria-pressed') === 'true'));
    else stop();
  });
  slider.addEventListener('input', () => stage.style.setProperty('--split', slider.value + '%'));
  button.addEventListener('click', () => timer ? stop() : play());
  if (location.hash === '#camera-comparison') panel.open = true;
})();
