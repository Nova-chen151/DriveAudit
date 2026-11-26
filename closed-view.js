const closedTabs = [...document.querySelectorAll('[data-loop-scene]')];
const closedSlots = [...document.querySelectorAll('[data-loop-policy]')];
const closedFilms = closedSlots.map(slot => slot.querySelector('video'));
const loopControls = document.querySelector('.loop-controls');
const loopPlay = document.querySelector('#loop-play-together');
const loopTime = document.querySelector('#loop-time');
const loopTimeValue = document.querySelector('#loop-time-value');
let comparePlayback = false;
const isZhLoop = () => document.documentElement.lang === 'zh-CN';
const syncPlayLabel = playing => (isZhLoop() ? (playing ? '全部暂停' : '同步播放') : (playing ? 'Pause both' : 'Play together'));

function stopComparison() {
  comparePlayback = false;
  closedFilms.forEach(video => video.pause());
  loopPlay.textContent = syncPlayLabel(false);
}

function setComparisonTime(seconds) {
  loopTime.value = String(seconds);
  loopTimeValue.textContent = seconds.toFixed(1) + ' s';
}

function showClosedScene(scene) {
  const { policies: videos, distance } = closedScenes[scene];
  stopComparison();
  setComparisonTime(0);
  closedTabs.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.loopScene === scene)));
  closedSlots.forEach((slot, i) => {
    const video = closedFilms[i];
    const status = slot.querySelector('.loop-status');
    const heading = slot.querySelector('h4');
    const policy = heading.firstChild.textContent.trim();
    const distanceLabel = slot.querySelector('.loop-distance');
    const url = videos[slot.dataset.loopPolicy];
    if (url) {
      if (video.getAttribute('src') !== url) video.src = url;
      video.setAttribute('aria-label', policy + ' DreamForge closed loop for ' + scene);
      video.hidden = false;
      status.hidden = true;
      if (distanceLabel) distanceLabel.textContent = 'Executed ' + distance[slot.dataset.loopPolicy].toFixed(1) + ' m in 6 s';
    } else {
      video.removeAttribute('src');
      video.load();
      video.hidden = true;
      status.textContent = 'Awaiting a verified rollout.';
      status.hidden = false;
      if (distanceLabel) distanceLabel.textContent = '';
    }
  });
  loopControls.hidden = closedFilms.some(video => video.hidden);
}

loopPlay.addEventListener('click', async () => {
  if (comparePlayback) {
    stopComparison();
    return;
  }
  const seconds = Number(loopTime.value);
  try {
    closedFilms.forEach(video => { video.currentTime = seconds; });
    comparePlayback = true;
    loopPlay.textContent = syncPlayLabel(true);
    await Promise.all(closedFilms.map(video => video.play()));
  } catch {
    stopComparison();
  }
});

loopTime.addEventListener('input', () => {
  stopComparison();
  const seconds = Number(loopTime.value);
  closedFilms.forEach(video => { video.currentTime = seconds; });
  setComparisonTime(seconds);
});

closedFilms[0].addEventListener('timeupdate', () => {
  if (!comparePlayback) return;
  const second = closedFilms[0].currentTime;
  if (Math.abs(closedFilms[1].currentTime - second) > 0.2) closedFilms[1].currentTime = second;
  setComparisonTime(Math.min(Number(loopTime.max), Math.floor(second * 2) / 2));
});
closedFilms.forEach(video => video.addEventListener('pause', () => {
  if (comparePlayback) stopComparison();
}));
closedTabs.forEach(tab => tab.addEventListener('click', () => showClosedScene(tab.dataset.loopScene)));
addEventListener('driveaudit-lang-change', () => {
  loopPlay.textContent = syncPlayLabel(comparePlayback);
});
showClosedScene(closedTabs[0].dataset.loopScene);
