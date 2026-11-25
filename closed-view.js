const closedTabs = [...document.querySelectorAll('[data-loop-scene]')];
const closedSlots = [...document.querySelectorAll('[data-loop-policy]')];

function showClosedScene(scene) {
  const videos = closedScenes[scene].policies;
  closedTabs.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.loopScene === scene)));
  closedSlots.forEach(slot => {
    const video = slot.querySelector('video');
    const status = slot.querySelector('.loop-status');
    const policy = slot.querySelector('h4').textContent;
    const url = videos[slot.dataset.loopPolicy];
    video.pause();
    if (url) {
      if (video.getAttribute('src') !== url) video.src = url;
      video.setAttribute('aria-label', policy + ' DreamForge closed loop for ' + scene);
      video.hidden = false;
      status.hidden = true;
    } else {
      video.removeAttribute('src');
      video.load();
      video.hidden = true;
      status.textContent = 'Awaiting a verified rollout.';
      status.hidden = false;
    }
  });
}

closedTabs.forEach(tab => tab.addEventListener('click', () => showClosedScene(tab.dataset.loopScene)));
