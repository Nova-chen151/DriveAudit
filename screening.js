(() => {
  const player = document.querySelector('#cut');
  const buttons = document.querySelectorAll('.cutlist button');
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  player.addEventListener('loadedmetadata', () => {
    player.currentTime = Math.min(4.5, player.duration / 2);
    if (!motion.matches) player.play().catch(() => {});
  });
  for (const button of buttons) button.addEventListener('click', () => {
    const {key, model, shot, caption} = button.dataset;
    player.poster = 'media/contrasts/' + key + '.jpg';
    player.src = 'media/contrasts/' + key + '.mp4';
    document.querySelector('#against').textContent = model;
    document.querySelector('#shot').textContent = shot;
    document.querySelector('#annotation').textContent = caption;
    buttons.forEach(b => b.setAttribute('aria-pressed', String(b === button)));
    if (!motion.matches) player.play().catch(() => {});
  });
  if (player.readyState >= 1 && !motion.matches) player.play().catch(() => {});
})();
