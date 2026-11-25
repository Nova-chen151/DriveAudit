const inspector = document.querySelector('.camera-inspect');
const stage = inspector.querySelector('.stage');
const slider = inspector.querySelector('#split');
const clips = ['left', 'right'].map(id => document.getElementById(id));
const choices = inspector.querySelectorAll('.choices button');
const toggle = inspector.querySelector('#camera-play');
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
let scene;
let generation = 0;

slider.addEventListener('input', () => stage.style.setProperty('--split', slider.value + '%'));
function pauseClips() {
  generation++;
  clips.forEach(video => video.pause());
  toggle.textContent = 'Play clips';
}
async function playClips() {
  const attempt = ++generation;
  try {
    await Promise.all(clips.map(video => video.play()));
    if (attempt === generation) toggle.textContent = 'Pause clips';
  } catch {
    if (attempt === generation) pauseClips();
  }
}
function selectClip(button) {
  if (scene !== button.dataset.scene) {
    pauseClips();
    scene = button.dataset.scene;
    clips.forEach((video, i) => {
      const model = i === 0 ? 'opendwm' : 'dreamforge';
      video.src = 'media/' + model + '-' + scene + '.mp4';
      video.preload = 'auto';
      video.load();
    });
  }
  stage.style.setProperty('--camera-offset', button.dataset.camera === 'CAM_FRONT' ? '-100%' : '0%');
  document.querySelector('#caption').textContent = button.dataset.caption;
  for (const choice of choices) choice.setAttribute('aria-pressed', String(choice === button));
  if (inspector.open && !reduceMotion.matches) playClips();
}
for (const button of choices) button.addEventListener('click', () => selectClip(button));
inspector.addEventListener('toggle', () => {
  if (inspector.open) selectClip(inspector.querySelector('[aria-pressed="true"]'));
  else pauseClips();
});
toggle.addEventListener('click', () => clips[0].paused ? playClips() : pauseClips());
clips[0].addEventListener('timeupdate', () => {
  if (!clips[0].paused && Math.abs(clips[0].currentTime - clips[1].currentTime) > .12)
    clips[1].currentTime = clips[0].currentTime;
});
clips[0].addEventListener('ended', () => {
  clips.forEach(video => { video.currentTime = 0; });
  if (inspector.open) playClips();
});
