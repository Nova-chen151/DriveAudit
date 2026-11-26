const panel = document.querySelector('#policy-explorer');
const policyTabs = [...panel.querySelectorAll('[data-policy]')];
const sceneTabs = [...panel.querySelectorAll('[data-policy-scene]')];
const policyVideo = panel.querySelector('#policy-film');
const activeLabel = panel.querySelector('#policy-active');
const supportLabel = panel.querySelector('#policy-support');
const metricRows = [...panel.querySelectorAll('.policy-scores tbody tr')];
const panelRows = [...panel.querySelectorAll('.policy-panel-scores tbody tr')];
let policyName = policyTabs[0].dataset.policy;
let sceneId = sceneTabs[0].dataset.policyScene;

function showPolicyPair() {
  const scene = policyResults[sceneId];
  const result = scene.policies[policyName];
  const videoPath = 'media/policy-fixed/' + result.file + '.mp4';
  if (policyVideo.getAttribute('src') !== videoPath) {
    const stayPlaying = !policyVideo.paused;
    const time = policyVideo.currentTime;
    policyVideo.src = videoPath;
    if (time > 0) {
      policyVideo.addEventListener('loadedmetadata', () => {
        if (!policyVideo.currentSrc.endsWith(videoPath)) return;
        policyVideo.currentTime = Math.min(time, policyVideo.duration);
        if (stayPlaying) policyVideo.play().catch(() => {});
      }, { once: true });
    }
  }
  policyVideo.setAttribute('aria-label', policyName + ' paired replay with predicted paths');
  if (activeLabel) activeLabel.textContent = policyName + ' / nuScenes ' + sceneId;
  if (supportLabel) supportLabel.textContent = scene.frames.length + ' decision windows shared across all six policies';
  metricRows.forEach((row, i) => {
    row.querySelectorAll('td').forEach((cell, model) => {
      cell.textContent = result.means[model][i].toFixed(3);
    });
  });
  panelRows.forEach(row => {
    const policy = row.querySelector('th').textContent;
    row.querySelectorAll('td').forEach((cell, model) => {
      cell.textContent = scene.policies[policy].means[model][0].toFixed(3);
    });
    if (policy === policyName) row.dataset.selected = '';
    else delete row.dataset.selected;
  });
}

policyTabs.forEach(button => button.addEventListener('click', () => {
  policyName = button.dataset.policy;
  policyTabs.forEach(tab => tab.setAttribute('aria-pressed', String(tab === button)));
  showPolicyPair();
}));
sceneTabs.forEach(button => button.addEventListener('click', () => {
  if (sceneId === button.dataset.policyScene) return;
  sceneId = button.dataset.policyScene;
  policyVideo.pause();
  policyVideo.currentTime = 0;
  sceneTabs.forEach(tab => tab.setAttribute('aria-pressed', String(tab === button)));
  showPolicyPair();
}));
showPolicyPair();
