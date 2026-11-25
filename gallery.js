const stage = document.querySelector('.stage');
const slider = document.querySelector('#split');
slider.addEventListener('input', () => stage.style.setProperty('--split', slider.value + '%'));
const choices = document.querySelectorAll('.choices button');
for (const button of choices) {
  button.addEventListener('click', () => {
    document.querySelector('#left').src = button.dataset.left;
    document.querySelector('#right').src = button.dataset.right;
    document.querySelector('#caption').textContent = button.dataset.caption;
    for (const choice of choices) choice.setAttribute('aria-pressed', String(choice === button));
  });
}
