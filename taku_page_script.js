// 打开弹幕的页面，然后复制粘贴下面代码即可

const targetElement = document.querySelector('uni-view.previewzoom.turnover.uncut');
if (targetElement) {
  const playSound = () => {
    const audio = new Audio('https://interactive-examples.mdn.mozilla.net/media/cc0-audio/t-rex-roar.mp3'); // Replace with your sound file URL
    audio.play().catch(error => console.error('Error playing sound:', error));
  };

  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      if (mutation.type === 'childList') {
        mutation.addedNodes.forEach((node) => {
          if (node.nodeType === Node.ELEMENT_NODE && node.tagName.toLowerCase() === 'uni-view') {
            playSound();
          }
        });
      }
    });
  });

  const config = { childList: true };
  observer.observe(targetElement, config);
  window.myObserver = observer;
} else {
  console.error('Target element not found.');
}
