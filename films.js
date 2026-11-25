const films = [...document.querySelectorAll('.filmgrid video')];
const scene = document.querySelector('#scene');
const renderer = document.querySelector('#renderer');
const pair = document.querySelector('#play');
let linked = false;
function stopPair() {
  linked = false;
  films.forEach(film => film.pause());
  pair.textContent = 'Play pair';
}
function updateFilms() {
  stopPair();
  ['opendwm', renderer.value].forEach((model, i) => {
    const stem = 'media/' + model + '-' + scene.value;
    films[i].src = stem + '.mp4';
    films[i].poster = stem + '-poster.jpg';
    films[i].load();
  });
  const name = renderer.selectedOptions[0].textContent;
  document.querySelector('#label1').textContent = name;
  films[1].setAttribute('aria-label', name + ' six-camera sequence');
}
scene.addEventListener('change', updateFilms);
renderer.addEventListener('change', updateFilms);
pair.addEventListener('click', async () => {
  if (linked) return stopPair();
  if (films[0].ended) films[0].currentTime = 0;
  films[1].currentTime = films[0].currentTime;
  linked = true;
  try {
    await Promise.all(films.map(film => film.play()));
    if (linked) pair.textContent = 'Pause pair';
  } catch { stopPair(); }
});
films.forEach(film => {
  film.addEventListener('pause', () => { if (linked && film.paused) stopPair(); });
});
films[0].addEventListener('timeupdate', () => {
  if (linked && Math.abs(films[1].currentTime - films[0].currentTime) > 0.12)
    films[1].currentTime = films[0].currentTime;
});
