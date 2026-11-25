const panel = document.querySelector('#policy-explorer');
const policyTabs = [...panel.querySelectorAll('[data-policy]')];
const pairTabs = [...panel.querySelectorAll('[data-pair]')];
const policyVideo = panel.querySelector('#policy-film');
const activeLabel = panel.querySelector('#policy-active');
const supportLabel = panel.querySelector('#policy-support');
const rendererLabel = panel.querySelector('#policy-renderer');
const metricRows = [...panel.querySelectorAll('.policy-scores tbody tr')];
let policyName = policyTabs[0].dataset.policy;

function showPolicyPair(index) {
  const options = policyResults[policyName];
  const result = options[index];
  const videoPath = 'media/policy-stack/' + result.file + '.mp4?v=shared-bev';
  if (policyVideo.getAttribute('src') !== videoPath) {
    policyVideo.pause();
    policyVideo.src = videoPath;
  }
  policyVideo.setAttribute('aria-label', policyName + ', OpenDWM and ' + result.renderer + ' camera inputs and predicted paths');
  activeLabel.textContent = policyName + ' / ' + result.dataset + ' ' + result.scene;
  supportLabel.textContent = result.frames.length + ' matched decision windows';
  rendererLabel.textContent = result.renderer;
  pairTabs.forEach((button, i) => {
    button.querySelector('span').textContent = options[i].dataset + ' ' + options[i].scene;
    button.querySelector('small').textContent = 'OpenDWM / ' + options[i].renderer;
    button.setAttribute('aria-pressed', String(i === index));
  });
  metricRows.forEach((row, i) => {
    [...row.querySelectorAll('td')].forEach((cell, model) => {
      cell.textContent = result.means[model][i].toFixed(3);
    });
  });
}
policyTabs.forEach(button => button.addEventListener('click', () => {
  policyName = button.dataset.policy;
  policyTabs.forEach(tab => tab.setAttribute('aria-pressed', String(tab === button)));
  showPolicyPair(0);
}));
pairTabs.forEach((button, i) => button.addEventListener('click', () => showPolicyPair(i)));
